#' This file takes a folder of images and trims the images to (mostly) the ball of the foot.
#' Then, the trimmed images are fed through the neural network to produce features.
#' Finally, a distance matrix is computed.

source_dir <- "~/Projects/CSAFE/ShoeScrapeR/extra/photos"
output_dir <- "~/Projects/CSAFE/ShoeScrapeR/extra/crop_square"

# --- Libraries ----------------------------------------------------------------
library(tidyverse)
library(keras)
library(furrr)
# ------------------------------------------------------------------------------

# --- Set up parallel env ------------------------------------------------------
plan(multicore)
# ------------------------------------------------------------------------------

# --- Sourced files ------------------------------------------------------------
source("~/models/shoe_nn/Generate_Model_Images.R")
source("~/models/shoe_nn/Image_Feature_Functions.R")
# ------------------------------------------------------------------------------

# --- Load model details -------------------------------------------------------
model_path <- "~/models/shoe_nn/TrainedModels/"

newest_model <- get_newest(dir = model_path, pattern = "weights.h5")
model_dir <- newest_model$path
load(list.files(model_dir, "-history.Rdata", full.names = T)[1])
load(file.path(get_newest()$path, get_newest(pattern = "\\d.Rdata")$base_file))

model_wts_file <- file.path(newest_model$path, newest_model$base_file)
loaded_model <- set_weights(model_wts_file)
# ------------------------------------------------------------------------------

# --- Work with images ---------------------------------------------------------
# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Crop nicely
crop_images(source_dir, output_dir)

# Get image statistics
img_stats <- debug_image(dir = output_dir)

# Get features
features <- furrr::future_map_dfr(get_folder_files(output_dir, pattern = ".jpg"),
                                  extract_features, model = loaded_model)

features <- features %>%
  mutate(partial_split = basename(img) %>%
           str_remove_all("_(product|color)_\\d{1,}") %>%
           str_remove_all("\\.jpg") %>%
           fix_brands(),
         brand = str_extract(partial_split, "[\\w]{1,}"),
         model_approx = str_remove(partial_split, paste0(brand, "-")) %>%
           str_extract("[\\w]{1,}"))

save(img_stats, features, file = file.path(output_dir, "Feature_Results.Rdata"))
