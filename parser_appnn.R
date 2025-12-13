#!/usr/bin/env Rscript
# Parser for data from APPNN () R package
# Usage: Rscript pipeline_appnn.R <input.fasta>

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript pipeline_appnn.R <input.fasta>\n")
  cat("Example: Rscript pipeline_appnn.R proteins.fasta\n")
  quit(status = 1)
}

input_fasta <- args[1]
intermediate_csv <- "output_proteins.csv"
output_dir <- "APPNN_parsed"

# --- STEP 1: APPNN Analysis (from APPNN_parser.R) ---

cat("Step 1: Running APPNN analysis...\n")

# Function from APPNN_parser.R
read_fasta_concatenated <- function(file_path) {
  file_lines <- readLines(file_path, warn = FALSE)
  sequences <- character()
  names <- character()
  current_seq <- ""
  current_name <- ""
  reading_sequence <- FALSE

  for (line in file_lines) {
    line <- trimws(line)

    if (nchar(line) == 0) {
      next
    }

    if (startsWith(line, ">")) {
      if (reading_sequence && nchar(current_seq) > 0) {
        sequences <- c(sequences, current_seq)
        names <- c(names, current_name)
      }
      current_name <- sub("^>", "", line)
      current_name <- trimws(current_name)
      current_seq <- ""
      reading_sequence <- TRUE
    } else if (reading_sequence) {
      clean_line <- gsub("[[:space:][:digit:]]", "", line)
      current_seq <- paste0(current_seq, clean_line)
    }
  }
  if (reading_sequence && nchar(current_seq) > 0) {
    sequences <- c(sequences, current_seq)
    names <- c(names, current_name)
  }
  return(data.frame(name = names, sequence = sequences, stringsAsFactors = FALSE))
}

process_fasta <- function(input_file, output_file) {
  cat(paste("  Reading FASTA file:", input_file, "\n"))
  
  fasta_data <- read_fasta_concatenated(input_file)
  
  if (nrow(fasta_data) == 0) {
    stop("No sequences found in the FASTA file.")
  }
  
  total_sequences <- nrow(fasta_data)
  cat(paste("  Found", total_sequences, "sequences\n"))
  
  df <- data.frame(
    sequence_name = character(total_sequences),
    sequence = character(total_sequences),
    overall = character(total_sequences),
    aminoacids = character(total_sequences),
    hotspots = character(total_sequences),
    stringsAsFactors = FALSE
  )
  
  sequences <- fasta_data$sequence
  cat("  Running APPNN...\n")
  
  tryCatch({
    # Load the appnn library
    if (!requireNamespace("appnn", quietly = TRUE)) {
      stop("appnn package is not installed. Please install it first.")
    }
    
    predictions <- appnn::appnn(sequences)
    cat("  Analysis completed\n")
    
    for (i in 1:total_sequences) {
      df[i, "sequence_name"] <- fasta_data$name[i]
      df[i, "sequence"] <- fasta_data$sequence[i]
      pred_i <- predictions[[i]]
      
      if (length(pred_i) >= 2 && length(pred_i[[2]]) > 0) {
        df[i, "overall"] <- toString(pred_i[[2]])
      } else {
        df[i, "overall"] <- NA
      }
      
      if (length(pred_i) >= 3 && length(pred_i[[3]]) > 0) {
        aa_scores <- pred_i[[3]]
        if (is.matrix(aa_scores)) {
          df[i, "aminoacids"] <- toString(as.vector(aa_scores))
        } else {
          df[i, "aminoacids"] <- toString(aa_scores)
        }
      } else {
        df[i, "aminoacids"] <- NA
      }
      
      if (length(pred_i) >= 4 && length(pred_i[[4]]) > 0) {
        hotspots <- pred_i[[4]]
        if (is.list(hotspots) && length(hotspots) > 0) {
          hotspot_str <- sapply(hotspots, function(x) paste(x, collapse = "-"))
          df[i, "hotspots"] <- paste(hotspot_str, collapse = ";")
        } else if (is.matrix(hotspots)) {
          df[i, "hotspots"] <- toString(hotspots)
        } else {
          df[i, "hotspots"] <- toString(hotspots)
        }
      } else {
        df[i, "hotspots"] <- NA
      }
    }
    
  }, error = function(e) {
    cat(paste("  Error running APPNN:", e$message, "\n"))
    for (i in 1:total_sequences) {
      df[i, "sequence_name"] <- fasta_data$name[i]
      df[i, "sequence"] <- fasta_data$sequence[i]
      df[i, "overall"] <- NA
      df[i, "aminoacids"] <- NA
      df[i, "hotspots"] <- NA
    }
  })
  
  cat(paste("  Writing intermediate results to:", output_file, "\n"))
  write.csv(df, output_file, row.names = FALSE)
  return(df)
}

# Run Step 1
tryCatch({
  intermediate_data <- process_fasta(input_fasta, intermediate_csv)
  cat("Step 1 complete!\n\n")
}, error = function(e) {
  cat(paste("Error in Step 1:", e$message, "\n"))
  quit(status = 1)
})

# --- STEP 2: Parse and Save Results (from CSV_APPNN_parser.R) ---

cat("Step 2: Parsing APPNN results...\n")

process_protein_data <- function(input_file) {
  data <- read_csv(input_file)
  all_proteins_list <- list()
  
  for (i in 1:nrow(data)) {
    sequence_name <- data$sequence_name[i]
    sequence <- data$sequence[i]
    aminoacids_scores <- as.numeric(unlist(strsplit(gsub("[\\[\\]\\n]", "", data$aminoacids[i]), ", ")))
    hotspots <- data$hotspots[i]
    aminoacids <- strsplit(sequence, "")[[1]]
    
    protein_df <- data.frame(
      sequence_name = sequence_name,
      aminoacid_position = 1:length(aminoacids),
      aminoacid = aminoacids,
      aminoacid_score = aminoacids_scores[1:length(aminoacids)],
      stringsAsFactors = FALSE
    )
    
    hotspot_ranges <- numeric()
    if (!is.na(hotspots) && hotspots != "") {
      ranges <- strsplit(hotspots, ";")[[1]]
      for (range_str in ranges) {
        range_parts <- strsplit(range_str, "-")[[1]]
        if (length(range_parts) == 2) {
          start_pos <- as.numeric(range_parts[1])
          end_pos <- as.numeric(range_parts[2])
          hotspot_ranges <- c(hotspot_ranges, start_pos:end_pos)
        }
      }
    }
    
    protein_df$hotspot_region <- ifelse(protein_df$aminoacid_position %in% hotspot_ranges, 1, 0)
    all_proteins_list[[sequence_name]] <- protein_df
  }
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
    cat(paste("  Created directory:", output_dir, "\n"))
  }
  
  # Save individual protein files
  for (protein_name in names(all_proteins_list)) {
    clean_name <- gsub("[^[:alnum:]_]", "_", protein_name)
    output_filename <- paste0(clean_name, "_APPNN.csv")
    output_path <- file.path(output_dir, output_filename)
    write_csv(all_proteins_list[[protein_name]], output_path)
    cat(paste("  Saved:", output_path, "\n"))
  }
  
  # Delete intermediate file
  if (file.exists(input_file)) {
    file.remove(input_file)
    cat(paste("  Deleted intermediate file:", input_file, "\n"))
  }
  
  return(bind_rows(all_proteins_list))
}

# Run Step 2
tryCatch({
  final_data <- process_protein_data(intermediate_csv)
  cat("Step 2 complete!\n\n")
}, error = function(e) {
  cat(paste("Error in Step 2:", e$message, "\n"))
  quit(status = 1)
})
# Final summary
cat("========================================\n")
cat("Parsing completed!\n")
cat("Input FASTA file:", input_fasta, "\n")
cat("Output directory:", output_dir, "\n")
cat("Number of proteins processed:", length(unique(final_data$sequence_name)), "\n")
cat("========================================\n")