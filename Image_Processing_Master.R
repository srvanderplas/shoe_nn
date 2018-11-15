#!/usr/bin/Rscript --no-save --no-restore
#
# For way too much info, add --verbose
# For logging, append
# ./script.R > RProcessedImagesLog.Rout 2>&1
# to invocation
#
# This script is supposed to:
# - Determine the last-modified date of any dependent files
# - Read images and annotations from their respective directories
# - Split images into labeled annotations
# - Save these images to a directory labeled with the last-modified file date
# - Randomly assign images to train/test/validation directories and then
#   create symlinks from image directory to train/test/validation

################################################################################
# Parameters
wd <- setwd("/models/shoe_nn/RProcessedImages")
image_dir <- "../LabelMe/Images/Shoes/"
annot_dir <- "../LabelMe/Annotations/Shoes/"
code_dir <- "../LabelMe/Code/"
split_prop <- c(.7, .15, .15)
classes <- c(
  "bowtie", "chevron", "circle", "line", "polygon",
  "quad", "star", "text", "triangle"
)
aug_multiple <- 1
epochs <- 15

################################################################################
# Packages
library(dplyr)
library(magrittr)
library(stringr)

################################################################################
# Get last modified date - annotations, code, images, and this R script.
annotation_files <- list.files(annot_dir, pattern = ".xml$", full.names = T)
annotation_modified <- file.mtime(annotation_files)

code_files <- list.files(code_dir, pattern = ".R$", full.names = T)
code_modified <- file.mtime(code_files)

image_files <- list.files(image_dir, pattern = ".jpg$", full.names = T)
annotated_image_files <- gsub(".jpg", "", basename(image_files)) %in%
  gsub(".xml", "", basename(annotation_files))
image_files <-  image_files[annotated_image_files]
image_modified <- file.mtime(image_files)

last_mod_date <- c(annotation_modified, code_modified, image_modified,
                   file.mtime("../Image_Processing_Master.R")) %>%
  max() %>%
  as.POSIXct(tz = "America/Chicago")

rm(annotation_modified, code_modified, image_modified, annotated_image_files,
   annotation_files, code_files, image_files)

################################################################################
# Check to see if new data computation is necessary

dirs <- list.dirs(recursive = F)
data_dirs <- dirs[grepl("\\d{8}-\\d{6}", dirs)]
if (length(data_dirs) > 0) {
  data_dir_date <- gsub(".*(\\d{8}-\\d{6}).*", "\\1", data_dirs) %>%
    as.POSIXct(., format = "%Y%m%d-%H%M%S", tz = "America/Chicago") %>%
    max(na.rm = T)
} else {
  data_dir_date <- as.Date("1900-01-01")
}

run_process_script <- data_dir_date <= last_mod_date
rm(dirs, data_dirs, data_dir_date)

################################################################################
# Set up Processing Environment

# Create directory if it doesn't exist
if (run_process_script) {
  process_dir <- format(last_mod_date, "%Y%m%d-%H%M%S")
  if (!dir.exists(process_dir)) {
    dir.create(process_dir)
  }
}

# Run processing script
source(file.path(code_dir, "Processing.R"))

system(sprintf("chmod -R 777 %s", process_dir))

################################################################################
# Split into training/test/validation sets

if (run_process_script) {
  # For now, list only files that haven't been split into pieces
  pattern <- "[a-z\\(\\)RE_]*-\\d{1,}-.*\\.jpg"
  annot_image_files <- list.files(file.path(process_dir, "images"),
                                  pattern = pattern, full.names = T)

  system(sprintf("chmod -R 766 %s", file.path(process_dir, "images")))

  dir <- c("train_all", "test", "validation")

  sapply(file.path(process_dir, dir), function(x) {
    if (dir.exists(x)) {
      system(sprintf("rm -rf %s", x))
    }
  })

  # Create directories
  sapply(file.path(process_dir, dir), dir.create)

  newdir <- sample(dir, length(annot_image_files), replace = T, prob = split_prop)
  file.symlink(from = file.path(getwd(), annot_image_files),
               to = file.path(getwd(), process_dir, newdir, basename(annot_image_files)))

  train_dir <- "train"
  if (dir.exists(file.path(getwd(), process_dir, train_dir))) {
    system(sprintf("rm -rf %s", file.path(process_dir, train_dir)))
  }

  dir.create(file.path(getwd(), process_dir, train_dir))
  # Sample training images to get approximately equal numbers of each class
  training_all <- list.files(file.path(process_dir, "train_all"), full.names = T)
  training_class <- data_frame(file = training_all) %>%
    mutate(class = purrr::map(basename(file), function(x) {
      str_extract(x, "^([a-z\\(\\)RE_]*)") %>%
        str_replace_all("\\([RE]\\)", "") %>%
        str_split("_", simplify = F)
    })) %>%
    tidyr::unnest() %>%
    tidyr::unnest() %>%
    group_by(file) %>%
    mutate(weight = 1/n()) %>%
    ungroup() %>%
    mutate(recode = !class %in% classes,
           class = ifelse(recode, "other", class))
  sample_threshold <- training_class %>%
    group_by(class) %>%
    summarize(n = n()) %>%
    magrittr::extract2("n") %>% quantile(.25) %>% round()
  training_class_sample <- training_class %>%
    group_by(class) %>%
    sample_n(size = sample_threshold, replace = T, weight = weight) %>%
    ungroup() %>%
    select(file) %>% extract2("file")

  training_class_sample <- training_class_sample %>% table %>% as.data.frame(stringsAsFactors = F) %>%
    rename(.data = ., file = `.`, weight = Freq)

  newfilename <- ifelse(training_class_sample$weight == 1, basename(training_class_sample$file),
                        paste0(training_class_sample$weight, "+", basename(training_class_sample$file)))



  file.symlink(from = file.path("/models", "shoe_nn", "RProcessedImages", str_replace(training_class_sample$file, "train_all", "images")),
               to = file.path(process_dir, train_dir, newfilename))
}

################################################################################
# Run Model

source("../vgg16_augment_onehot_automation.R")
