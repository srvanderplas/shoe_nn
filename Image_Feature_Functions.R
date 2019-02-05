#' Crop Shoe to Center
#'
#' Crops shoe image, removing white space around the shoe and an additional
#' portion of the outer part of the shoe.
#' Returns an image with 500 px of height.
#'
#' Parameter documentation for imagemagick arguments can be found here:
#' \url{https://imagemagick.org/script/command-line-options.php}
#'
#' @param file input file - image of a shoe
#' @param outfile output image file - square image
#' @param fuzz "fuzz" in percent (imagemagick trim argument) to use for initial
#'          trimming
#' @param gravity Gravity to use for crop (default to East). See
#'          imagemagick documentation for other options.
#' @param crop A string compatible with imagemagick crop arguments
#'          (usually wxl+v+h, where w is width, l is length, h is horizontal offset, v is vertical offset)
crop_to_center <- function(file, outfile, fuzz = 4, gravity = "East", crop = "0x500+0+0") {
  command <- sprintf("convert %s -trim -fuzz %d%% -trim +repage -gravity %s -crop %s %s", file, fuzz, gravity, crop, outfile)
  output <- try(system(command, intern = T))
  list(command = command, output = output, outfile = outfile)
}

#' Crop to Ball of Shoe
#'
#' Crops shoe image to focus on the right side of the image (typically the ball
#' of the shoe if the shoe is aligned with the heel to the left side)
#'
#' Parameter documentation for imagemagick arguments can be found here:
#' \url{https://imagemagick.org/script/command-line-options.php}
#'
#' @param file input file - image of a shoe
#' @param outfile output image file - square image
#' @param gravity gravity to use for crop (default center).
#' @param crop_pct percent of the image to keep (width, height)
#' @param offset_pct percent of the image to use as offset (width, height)
#' @param output_size output size in pixels ("256x256")
#' @param extra_operations imagemagick commands to use on the tail end of the
#'          crop operation to perform color adjustment or other operations.
#'          Must be escaped properly for R + system - e.g. \\ for special chars.
#'          Can be set to 'normalize' as shorthand for
#'          "\\( +clone -equalize \\) -average"
crop_square_right <- function(file, outfile, gravity = "center", crop_pct = c(.8, .8), offset_pct = c(1/6, 0),
                            output_size = "256!x256!", extra_operations = "") {
  dims <- system(sprintf("identify %s", file), intern = T) %>%
    gsub("(.*?) JPEG (\\d{1,}x\\d{1,}) (.*)", "\\2", .) %>%
    stringr::str_split("x", n = 2, simplify = T) %>%
    as.numeric()

  if (extra_operations == "normalize") extra_operations <- " \\( +clone -equalize \\) -average"

  command <- sprintf("convert %s -colorspace RGB -gravity %s -crop '%dx%d+%d+%d' -resize %s %s %s",
                  file, gravity,
                  round(dims[2]*crop_pct[1]), round(dims[2]*crop_pct[2]),
                  round(dims[1]*offset_pct[1]), round(dims[2]*offset_pct[2]),
                  output_size,
                  extra_operations,
                  outfile)
  output <- try(system(command, intern = T))
  return(list(command = command, output = output, outfile = outfile))
}

#' Crop Shoe Image
#'
#' Crops shoe image into a square region that should focus on the ball of the
#' foot if the image is aligned with the heel to the left and the toe to the
#' right.
#'
#' Parameter documentation for imagemagick arguments can be found here: \url{https://imagemagick.org/script/command-line-options.php}
#'
#' @param file input file - image of a shoe
#' @param outfile output image file - square image
crop_square_toe <- function(file, outfile, crop_center_args = list(), crop_square_right_args = list()) {
  td <- file.path(dirname(outfile), "temp")
  if (!dir.exists(td)) dir.create(td)
  tf <- file.path(td, basename(outfile))

  cc_args <- vector("list", length(crop_center_args) + 2)

  if (length(crop_center_args) > 0) {
    cc_args[3:(length(crop_center_args) + 2)] <- crop_center_args
  }
  cc_args[1:2] <- list(file = file, outfile = tf)

  cc_res <- do.call(crop_to_center, cc_args)

  csr_args <- vector("list", length(crop_square_right_args) + 2)

  if (length(crop_square_right_args) > 0) {
    csr_args[3:(length(crop_square_right_args) + 2)] <- crop_square_right_args
  }
  csr_args[1:2] <- list(file = tf, outfile = outfile)

  csr_res <- do.call(crop_square_right, csr_args)

  list(file = file, outfile = outfile, crop_center_res = cc_res, crop_square_right_res = csr_res)
}

#' Generate Cropped Images
#'
#' @param source_dir Source directory for images
#' @param output_dir Output directory for images
#' @param fun Function to crop images
#' @param clear_output_dir Remove files from output directory first?
#' @import
crop_images <- function(source_dir, output_dir, fun = crop_square_toe, clear_output_dir = T, ...) {
  imgs <- list.files(source_dir)
  if (clear_output_dir) file.remove(list.files(output_dir, full.names = T))
  furrr::future_walk2(file.path(source_dir, imgs), file.path(output_dir, imgs), fun, ...)
}

#' Extract features from image
#'
#' @param img_path Path to image
#' @param model Keras model used to calculate output features
#' @param classes classes used to train the model (for labeling the resulting data frame)
#' @importFrom keras predict.keras.engine.training.Model
extract_features <- function(img_path, model, classes) {

  img <- jpeg::readJPEG(img_path)

  try(dim(img) <- c(1, 256, 256, 3))
  if (!"try-error" %in% class(img)) {
    features <- model %>% predict(img)
  }

  tibble::as_tibble(features) %>%
    magrittr::set_colnames(default_classes) %>%
    dplyr::mutate(file_path = dirname(img_path),
                  img = basename(img_path))
}

#' Get image information (debugging)
#'
#' @param dir directory of images
#' @param img image file
debug_image <- function(dir = NULL,  img = NULL) {
  stopifnot(!is.null(dir) | !is.null(img))
  if (!is.null(dir) & !is.null(img)) {
    warning("img and dir both supplied; img argument will be ignored")
  }

  if (!is.null(dir)) {
    tf <- tempfile(fileext = ".txt")

    command <- sprintf("find %s -type f | parallel identify > %s", dir, tf)
    output <- try(system(command, intern = T))
    res <- sort(readLines(tf))

    files <- stringr::str_extract(res, "(^\\S{1,})")
    lw <- stringr::str_extract(res, "\\d{1,}x\\d{1,} ")
    color <- stringr::str_extract(res, "(sRGB|CMYK)")

    list(result = tibble::tibble(file = files, dims = lw, colorspace = color, full = res),
         command = command, output = output)
  } else if (!is.null(img)) {
    command <- paste0("identify ", img)
    output <- try(command, intern = T)
    res <- output

    files <- stringr::str_extract(res, "(^\\S{1,})")
    lw <- stringr::str_extract(res, "\\d{1,}x\\d{1,} ")
    color <- stringr::str_extract(res, "(sRGB|CMYK)")

    list(result = tibble::tibble(file = files, dims = lw, colorspace = color, full = res),
         command = command, output = output)
  }
}

#' Clean shoe model names
#'
#' This function gets rid of auxiliary dashes in brand names of image files.
#' For example, "dr-marten-model-name-style" becomes "drmarten-model-name-style"
#' This is just an alias for str_replace with a specific pattern/replacement
#' collection
#' @param x vector of image names
fix_brands <- function(x) {
  multiword_models <- c("dr-mart" = "drmart", "1-state" = "1state", "5-11" = "511", "a2-by-aerosoles" = "aerosoles", "b-52-by-bullboxer" = "bullboxer",
                        "b-o-c" = "boc", "cc-corso-como" = "cccorsocomo", "dr-scholls" = "drscholls", "dv-by-dolce" = "dolce",
                        "ed-ellen-degeneres" = "ellendegeneres", "el-naturalista" = "elnaturalista", "g-by-guess" = "guess", "g-h-bass-co" = "ghbassco",
                        "hi-tec" = "hitec", "j-m-collection" = "jmcollection", "j-m-est-1850" = "jmcollection", "j-renee" = "jrenee",
                        "j-slides" = "jslides", "k-jacques" = "kjacques", "k-swiss" = "kswiss", "l-b-evans" = "lbevans", "l-k-bennett" = "lkbennett",
                        "la-canadienne" = "lacanadienne", "la-sportiva" = "lasportiva", "m-a-p" = "map", "m-f-western" = "mfwestern", "me-too" = "metoo",
                        "on-cloud-2-0" = "oncloud", "on-cloud" = "oncloud", "pf-flyers" = "pfflyers", "to-boot-new-york" = "tobootnewyork",
                        "ys-by-yohji-yamamoto" = "yohjiyamamoto", "z-zegna" = "zzegna", "five-ten" = "fiveten")
  stringr::str_replace_all(x, multiword_models)
}

#' Get folder files
#'
#' This is a very simple function that lists files in a directory while including
#' only the path to the directory in the file path.
#' @param dir directory
#' @param ...
get_folder_files <- function(dir, ...) file.path(dir, list.files(path = dir, include.dirs = F, ...))

