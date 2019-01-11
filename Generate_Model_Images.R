library(magrittr)
library(lubridate)
library(stringr)
library(jpeg)
library(keras)
library(imager)
library(tidyr)
library(magick)
library(viridis)

# --- Variables ----------------------------------------------------------------
default_classes <- c(
  "bowtie", "chevron", "circle", "line", "polygon",
  "quad", "star", "text", "triangle"
)

# --- Functions ----------------------------------------------------------------

# Get most recent model
get_newest <- function(dir = "~/models/shoe_nn/TrainedModels/", pattern = "fullimage.rdata") {
  dirlist <- list.dirs(path = dir) %>% rev()
  newest_file <- NULL
  startidx <- 1:10
  while (is.null(newest_file) & max(startidx) <= length(dirlist)) {
    newest_file <- dirlist %>%
      magrittr::extract(startidx) %>%
      sapply(function(x) list.files(x, pattern = pattern, recursive = F, full.names = T)) %>%
      unlist() %>%
      as.character()

    mtimes <- file.mtime(newest_file)

    newest_file <- newest_file[which.max(mtimes)]

    startidx <- pmin(max(startidx) + 1:10, length(dirlist))
  }

  path <- dirname(newest_file)
  base_file <- basename(newest_file)
  start_date <- str_extract(base_file, "\\d{4}-\\d{2}-\\d{2}[_ ]\\d{2}:\\d{2}:\\d{2}")
  prefix <- str_remove(base_file, "-history.Rdata|-weights.h5|.h5|.Rdata|fullimage.rdata") %>%
    str_remove(start_date) %>%
    str_remove("^_")
  process_dir <- basename(path)
  return(list(path = path, base_file = base_file, prefix = prefix, start_date = start_date, process_dir = process_dir))
}

set_weights <- function(model_wts_file) {
  input <- layer_input(shape = c(256, 256, 3))

  conv_base <- application_vgg16(
    weights = "imagenet",
    include_top = FALSE,
    input_tensor = input)

  output <- conv_base$output %>%
    layer_flatten(input_shape = input) %>%
    layer_dense(units = 256,
                activation = "relu",
                input_shape = 8 * 8 * 512,
                name = "dense_1") %>%
    layer_dropout(rate = 0.5,
                  name = "dropout_1") %>%
    layer_dense(units = 9,
                activation = "sigmoid",
                name = "dense_2")

  model <- keras_model(input, output)

  load_model_weights_hdf5(model, model_wts_file, by_name = T)
}

calc_heatmap <- function(img_path, model, classes = default_classes, scale_by_prob = F) {
  img <- jpeg::readJPEG(img_path)
  dim(img) <- c(1, dim(img))

  predictions <- model %>%
    predict(img, verbose = T) %>%
    as.vector() %>%
    set_names(classes)

  # round(predictions,3)

  true_labels <- sapply(classes, function(x){grepl(x, basename(img_path))}) %>%
    as.numeric()

  n_classes <- length(classes)
  heatmap <- array(dim = c(n_classes, 16, 16))
  successful_heatmap <- c()

  for (j in 1:n_classes) {
    scalep <- ifelse(scale_by_prob, predictions[j], 1)
    # Make 16x16 heatmap matrix for each class
    img_output <- model$output[,j]

    last_conv_layer <- model %>% get_layer("block5_conv3")
    grads <- k_gradients(img_output, last_conv_layer$output)[[1]]
    pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
    iterate <- k_function(list(model$input),
                          list(pooled_grads, last_conv_layer$output[1,,,]))
    c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
    for (i in 1:512) {
      conv_layer_output_value[,,i] <-
        conv_layer_output_value[,,i] * pooled_grads_value[[i]]
    }
    heatmap[j,,] <- apply(conv_layer_output_value, c(1,2), mean)
    heatmap[j,,] <- pmax(heatmap[j,,], 0)
    heatmap[j,,] <- heatmap[j,,] / max(heatmap[j,,]) * scalep

    # Check if the heatmap matrix contains any NaN values
    if (!anyNA(heatmap[j,,])) {
      successful_heatmap <- c(successful_heatmap, j)
    }
  }

  # heatmap <- heatmap/max(heatmap)

  if (is.null(successful_heatmap)) {
    message("No heatmaps were successful")
    return(list(img = img, heatmap = NULL, successful_heatmap = NULL, predictions = predictions, truth = true_labels))
  } else {
    return(list(img = img, heatmap = heatmap,
                successful_heatmap = successful_heatmap,
                predictions = predictions, truth = true_labels, classes = classes,
                img_path = img_path))
  }
}

heatmap_overlay <- function(heatmap, geometry = geometry, width = 256, height = 256, tdd = tempdir(),
                          bg = "white", col = terrain.colors(12)) {
  overlay_file <- tempfile(tmpdir = tdd, fileext = "png")
  png(overlay_file, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col, zlim = c(0, 1))
  dev.off()
  par(op)
  on.exit(file.remove(overlay_file), add = T)
  image_read(overlay_file) %>%
    image_resize(geometry, filter = "quadratic")
}

create_composite <- function(heatmap_data, save_file = F, outdir = ".", td = tempdir(), fixed_labels = T,
                             fail_file = file.path("/models/shoe_nn/", "poop.jpg")) {

  # Fix image dimensions
  dim(heatmap_data$img) <- dim(heatmap_data$img)[-1]
  image <- image_read(heatmap_data$img)

  # Turn successful heatmap arrays into overlays and apply to original image
  info <- image_info(image)
  geometry <- sprintf("%dx%d!", info$width, info$height)

  pal <- col2rgb(viridis(20), alpha = TRUE)
  alpha <- floor(seq(0, 255, length = ncol(pal)))
  pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
  correct_pal <- colorRampPalette(c("white", "cornflowerblue"))
  incorrect_pal <- colorRampPalette(c("white", "grey40"))

  n_classes <- length(heatmap_data$classes)

  blank_img <- image_blank(300, 300, color = "white")

  filelist <- rep(blank_img, n_classes)
  unmod_image <- image_composite(blank_img, image, offset = "+22+6") %>%
    # Add label
    image_annotate("Original Image", color = "black", size = 30, location = "+0+3", gravity = "South")

  for (j in 1:n_classes) {

    label <- paste(tools::toTitleCase(heatmap_data$classes[j]),
                   round(heatmap_data$predictions[j], 3), sep = ": ")
    intensity <- 100*round(heatmap_data$predictions[j], 2) + 1

    bg_col <- ifelse(heatmap_data$truth[j],
                     correct_pal(131)[30 + intensity],
                     incorrect_pal(101)[intensity])
    label_col <- ifelse((intensity > 60 && !heatmap_data$truth[j]), "white", "black")

    overlay <- heatmap_overlay(heatmap_data$heatmap[j,,], geometry = geometry, tdd = td,
                               width = 14, height = 14, bg = NA, col = pal_col)

    if (j %in% heatmap_data$successful_heatmap) {
      # Create blank image with correct bg color
      filelist[j] <- image_blank(300, 300, color = bg_col) %>%
        # Add label
        image_annotate(label, color = label_col,
                       size = 30, location = "+0+3", gravity = "South") %>%
        # add image
        image_composite(image, offset = "+22+6") %>%
        # add overlay
        image_composite(overlay, offset = "+22+6")
    } else {
      filelist[j] <- blank_img %>%
        image_annotate("Failed\nto\ngenerate", color = "black", size = 30, gravity = "Center")
    }
  }

  composite_image <-   c(
    image_append(c(blank_img, unmod_image, blank_img), stack = T),
    image_append(filelist[1:3], stack = T),
    image_append(filelist[4:6], stack = T),
    image_append(filelist[7:9], stack = T)
  ) %>%
    image_append()

  if (save_file) {
    final_save_file <- file.path(outdir,
                                 sprintf("heatmap-%s.png", tools::file_path_sans_ext(basename(heatmap_data$img_path))))

    composite_image %>%
      image_write(path = final_save_file, format = "png")
    return(final_save_file)
  }

  return(composite_image)
}

# get_newest()
# get_newest(pattern = "history")
# newest_files <- get_newest()
# image_dir <- file.path("/models/shoe_nn/RProcessedImages", newest_files$process_dir)
#
# # File containing weights from model
# model_wts_file <- get_newest(pattern = "weights")
# model_wts_file <- file.path(model_wts_file$path, model_wts_file$base_file)
#
# test_images <- list.files(file.path(image_dir, "test"), "*.jpg", full.names = T)
# loaded_model <- set_weights(model_wts_file)
#
