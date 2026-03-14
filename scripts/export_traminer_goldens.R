#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) > 0) args[[1]] else "tests/goldens/traminer_reference.json"

suppressPackageStartupMessages({
  library(TraMineR)
  library(jsonlite)
})

extract_fixture <- function(frame, cols) {
  wide <- frame[, cols]
  seqs <- seqdef(wide)
  trate <- seqcost(seqs, method = "TRATE")
  om <- seqdist(seqs, method = "OM", sm = trate$sm, indel = trate$indel)
  statd <- seqstatd(seqs)
  reps <- seqrep(seqs, criterion = "freq", diss = seqdist(seqs, method = "LCS"))
  list(
    wide = unname(as.matrix(wide)),
    columns = colnames(wide),
    trate_costs = unname(trate$sm),
    om_distances = unname(as.matrix(om)),
    state_distribution = unname(statd$Frequencies),
    entropy = unname(statd$Entropy),
    representatives = as.integer(attr(reps, "Index")) - 1
  )
}

datasets <- list(
  actcal = extract_fixture(actcal, 13:24),
  biofam = extract_fixture(biofam[1:12, ], 10:25),
  ex1 = extract_fixture(ex1, 1:13),
  mvad = extract_fixture(mvad[1:12, ], 17:30)
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(list(datasets = datasets), output, auto_unbox = TRUE, digits = NA, pretty = TRUE)

