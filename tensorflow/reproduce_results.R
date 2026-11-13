# Run from tensorflow/: Rscript reproduce_results.R
# The same native R cells as Restart Kernel and Run All (default: no training).
# IRkernel supplies rich display in Jupyter; this command also prints diagnostics.
nb <- jsonlite::fromJSON("README.ipynb", simplifyVector=FALSE)
for(cell in nb$cells) if(identical(cell$cell_type,"code")) {
  code <- paste0(unlist(cell$source),collapse="")
  eval(parse(text=code), envir=.GlobalEnv)
  # The terminal has no rich-display channel. Print the same recorded table
  # values; images remain in the adjacent notebook/HTML outputs.
  if(isTRUE(cell$metadata$embedded_data)) recorded_cell <- function(index) {
    cat("Recorded benchmark output; no new training.\n")
    for(out in recorded_outputs[[as.character(index)]]) {
      if(identical(out$output_type,"stream")) cat(paste0(unlist(out$text),collapse=""))
      else if(!is.null(out$data[["text/plain"]])) cat(paste0(unlist(out$data[["text/plain"]]),collapse=""),"\n")
      else if(!is.null(out$data[["text/html"]])) {
        text<-paste0(unlist(out$data[["text/html"]]),collapse="")
        cat(gsub("<[^>]*>"," ",gsub("</tr>","\n",text,fixed=TRUE)),"\n")
      } else if(!is.null(out$data[["image/png"]])) cat("Figure: see notebook or HTML.\n")
    }
  }
}
