library(magrittr)
library(lubridate)
library(stringr)
library(jpeg)
library(keras)

use_backend("tensorflow")
# install_keras()
if (!exists("classes")) {
  classes <- c(
    "bowtie", "chevron", "circle", "line", "polygon",
    "quad", "star", "text", "triangle"
  )
}

if (!exists("process_dir")) {
  process_dir <- list.files("/models/shoe_nn/RProcessedImages") %>%
    as_datetime() %>%
    max(na.rm = T) %>%
    gsub("[^0-9\\ ]", "", .) %>%
    gsub(" ", "-", .)
} else {
  process_dir <- file.path("/models/shoe_nn/RProcessedImages", process_dir)
}

if (!exists("aug_multiple")) {
  aug_multiple <- 3
}

if (!exists("epochs")) {
  epochs <- 15
}

process_dir <- gsub("[[:punct:]]models[[:punct:]]shoe_nn[[:punct:]]RProcessedImages[[:punct:]]{1,}", "\\1", process_dir)
process_dir <- gsub("^[/\\\\]{1,}", "", process_dir)


work_dir <- "/models/shoe_nn/TrainedModels"
start_date <- Sys.time() %>% gsub(" ", "_", .)
model_dir <- file.path(work_dir, process_dir)
dir.create(model_dir)

name_file <- function(date, ext) {
  pretrained_base <- "vgg16"
  mod_type <- "onehotaug"
  nclass <- paste0(length(classes), "class")
  pixel_size <- "256"

  filename <- paste(date, pretrained_base, mod_type, nclass,
    pixel_size,
    sep = "_"
  )

  file.path(model_dir, filename) %>%
    paste0(., ext)
}


base_dir <- file.path("/models/shoe_nn/RProcessedImages", process_dir)
train_dir <- file.path(base_dir, "train")
train_aug_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))

img_names <- list.files(train_dir) %>% str_remove(., "\\.jpg")
img_loc <- list.files(train_dir, full.names = T)

augment_img <- function(filename, times = 3) {
  # Determine number of times augmentation should happen
  rep_num <- str_extract(basename(filename), "^\\d") %>% as.numeric()

  if (!is.na(rep_num)) {
    file_dir <- dirname(filename)
    base_file_name <- file.path(file_dir, str_remove(basename(filename), "^\\d{1,}\\+"))
    file.rename(filename, base_file_name)
    newfilepath <- base_file_name
    newfilename <- basename(base_file_name) %>% str_remove("\\.jpg")
  } else {
    rep_num <- 1
    newfilepath <- filename
    newfilename <- basename(filename) %>% str_remove("\\.jpg")
  }

  img <- readJPEG(newfilepath)
  dim(img) <- c(1, dim(img))

  aug_generator <- image_data_generator(
    samplewise_std_normalization = T,
    rotation_range = 40,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 60,
    zoom_range = 0.1,
    channel_shift_range = .1,
    zca_whitening = T,
    vertical_flip = T,
    horizontal_flip = TRUE
  )

  images_iter <- flow_images_from_data(
    x = img, y = NULL,
    generator = aug_generator,
    batch_size = 1,
    save_to_dir = train_aug_dir,
    save_prefix = paste("aug", newfilename, sep = "_"),
    save_format = "jpg"
  )

  iter_num <- times
  while (rep_num > 0) {
    while (iter_num > 0) {
      reticulate::iter_next(images_iter)
      iter_num <- iter_num - 1
    }
    rep_num <- rep_num - 1
  }
}

for (i in list.files(train_dir, full.names = T)) {
  augment_img(i, times = aug_multiple)
}


get_labs <- function(directory, verbose = F) {

  sample_count <- length(list.files(directory))
  labels <- array(0, dim = c(sample_count, length(classes)))
  files <- list.files(directory)

  for (i in 1:sample_count) {
    if (verbose) cat(paste(i, ", ", sep=""))

    fname <- files[i]
    str <- substr(fname, 1, regexpr("-",fname)-1)
    for (j in 1:length(classes)) {
      labels[i, j] <- grepl(classes[j], str)
    }
  }
  labels
}

dir_to_data <- function(directory, verbose = F) {
  files <- list.files(directory, full.names = T)
  cat("Directory length:", length(files), "\n")

  if (verbose) {cat("Getting labels...\n")}
  labels <- get_labs(directory)

  if (verbose) {cat("Gathering data...\n")}
  data <- array(0, dim = c(length(files), 256, 256, 3))

  for (i in 1:length(files)) {
    if (verbose) {cat(i, ", ", sep = "")}
    img <- imager::load.image(files[i])
    dim(img) <- c(1, 256, 256, 3)
    data[i,,,] <- img
  }

  data <- data/255

  list(
    data = data,
    labels = labels
  )
}


train <- dir_to_data(train_dir, verbose = T)
validation <- dir_to_data(validation_dir, verbose = T)
test <- dir_to_data(test_dir, verbose = T)



input <- layer_input(shape = c(256, 256, 3))

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_tensor = input)

output <- conv_base$output %>%
  layer_flatten(input_shape = input) %>%
  layer_dense(units = 256, activation = "relu",
    input_shape = 8 * 8 * 512
  ) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = length(classes), activation = "sigmoid")

model <- keras_model(input, output)

freeze_weights(model, from = 1, to = 'block5_conv3')

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$data, train$labels,
  epochs = epochs,
  batch_size = 20,
  validation_data = list(validation$data, validation$labels)
)


png(filename = name_file(start_date, ".png"), width = 1000, height = 1000, type = "cairo", pointsize = 16)
plot(history)
dev.off()

save_model_hdf5(model, name_file(start_date, ".h5"))

preds <- model %>% predict(test$features)
test_labs <- test$labels
colnames(preds) <- colnames(test_labs) <- classes

save(classes, preds, test_labs, file = name_file(start_date, ".Rdata"))
base::save.image(name_file(start_date, "fullimage.rdata"))

tfdeploy::export_savedmodel(model, name_file(start_date, ""))
