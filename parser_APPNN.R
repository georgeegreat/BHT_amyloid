df <- data.frame(
  sequence = character(7),
  overall = character(7),
  aminoacids = character(7),
  hotspots = character(7),
  stringsAsFactors = FALSE
  )
for(i in 1:length(predictions)) {
  for(j in 1:4) {
    if(length(predictions[[i]][[j]]) > 0) {
      df[i, j] <- toString(predictions[[i]][[j]])
      } else {
        df[i, j] <- NA
      }
  }
  }
write.csv(df, "output.csv", row.names = FALSE)
library(readr)
output <- read_csv("output.csv", col_types = cols(aminoacids = col_character(), 
                                                  hotspots = col_character()))