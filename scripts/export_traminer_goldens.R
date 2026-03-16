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
  use_missing <- any(is.na(as.matrix(wide)))
  if (is.null(weights)) {
    seqs <- seqdef(wide)
  } else {
    seqs <- seqdef(wide, weights = weights)
  }
  trate <- seqcost(seqs, method = "TRATE", with.missing = use_missing)
  om <- seqdist(seqs, method = "OM", sm = trate$sm, indel = trate$indel, with.missing = use_missing)
  ham <- tryCatch(
    seqdist(seqs, method = "HAM", with.missing = use_missing),
    error = function(e) NULL
  )
  lcs_auto <- seqdist(seqs, method = "LCS", norm = "auto", with.missing = use_missing)
  statd <- seqstatd(seqs, with.missing = use_missing)
  trate_rates <- seqtrate(seqs, with.missing = use_missing)
  meant <- seqmeant(seqs, with.missing = use_missing)
  lcs <- seqdist(seqs, method = "LCS", with.missing = use_missing)
  reps <- seqrep(seqs, criterion = "freq", diss = lcs)
  state_order <- rownames(trate$sm)
  missing_state <- if (use_missing && length(state_order) > 0) tail(state_order, 1) else NULL
  list(
    wide = unname(as.matrix(wide)),
    columns = colnames(wide),
    weights = if (is.null(weights)) NULL else unname(as.numeric(weights)),
    with_missing = use_missing,
    states = unname(state_order),
    missing_state = if (is.null(missing_state)) NULL else unname(missing_state),
    trate_costs = unname(trate$sm),
    trate_indel = unname(trate$indel),
    om_distances = unname(as.matrix(om)),
    ham_distances = if (is.null(ham)) NULL else unname(as.matrix(ham)),
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
    schema_version = 5,
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
