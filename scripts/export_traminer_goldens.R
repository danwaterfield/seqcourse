#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) > 0) args[[1]] else "tests/goldens/traminer_reference.json"

suppressPackageStartupMessages({
  library(TraMineR)
  library(jsonlite)
})

load_traminer_dataset <- function(name) {
  data(list = name, package = "TraMineR", envir = environment())
  if (!exists(name, inherits = FALSE)) {
    stop(sprintf("TraMineR dataset '%s' could not be loaded.", name))
  }
  get(name, inherits = FALSE)
}

extract_fixture <- function(frame, cols, weights = NULL) {
  wide <- frame[, cols]
  if (is.null(weights)) {
    seqs <- seqdef(wide)
  } else {
    seqs <- seqdef(wide, weights = weights)
  }
  trate <- seqcost(seqs, method = "TRATE")
  om <- seqdist(seqs, method = "OM", sm = trate$sm, indel = trate$indel)
  ham <- seqdist(seqs, method = "HAM")
  lcs_auto <- seqdist(seqs, method = "LCS", norm = "auto")
  statd <- seqstatd(seqs)
  trate_rates <- seqtrate(seqs)
  meant <- seqmeant(seqs)
  reps <- seqrep(seqs, criterion = "freq", diss = seqdist(seqs, method = "LCS"))
  list(
    wide = unname(as.matrix(wide)),
    columns = colnames(wide),
    weights = if (is.null(weights)) NULL else unname(as.numeric(weights)),
    trate_costs = unname(trate$sm),
    trate_indel = unname(trate$indel),
    om_distances = unname(as.matrix(om)),
    ham_distances = unname(as.matrix(ham)),
    lcs_auto = unname(as.matrix(lcs_auto)),
    transition_rates = unname(trate_rates),
    mean_time = unname(meant),
    state_distribution = unname(statd$Frequencies),
    entropy = unname(statd$Entropy),
    representatives = as.integer(attr(reps, "Index")) - 1,
    representative_groups = as.integer(attr(reps, "Rep.group")) - 1
  )
}

actcal <- load_traminer_dataset("actcal")
biofam <- load_traminer_dataset("biofam")
ex1 <- load_traminer_dataset("ex1")
mvad <- load_traminer_dataset("mvad")

datasets <- list(
  actcal = extract_fixture(actcal, 13:24),
  biofam = extract_fixture(biofam[1:12, ], 10:25),
  ex1 = extract_fixture(ex1, 1:13, weights = ex1$weights),
  mvad = extract_fixture(mvad[1:12, ], 17:30)
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(
  list(
    schema_version = 2,
    upstream = list(
      package = "TraMineR",
      version = as.character(packageVersion("TraMineR"))
    ),
    datasets = datasets
  ),
  output,
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)
