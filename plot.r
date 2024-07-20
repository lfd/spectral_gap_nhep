#!/usr/bin/env Rscript

library(tidyverse)
library(ggh4x)
library(stringr)
source("layout.r")

create_save_locations()

d <- read.csv("results/MAXCUT/spectral_gap_evolution.csv",
              stringsAsFactors = FALSE)
d$fraction <- as.numeric(d$fraction)
d$gs <- as.numeric(d$gs)
d$fes <- as.numeric(d$fes)
d$gap <- as.numeric(d$gap)
d$num_qubits <- as.numeric(d$num_qubits)
print(d)

g <- ggplot(d) +
    geom_line(aes(x = fraction, y = gs), colour = COLOURS.LIST[[1]]) +
    geom_line(aes(x = fraction, y = fes), colour = COLOURS.LIST[[2]]) +
    facet_grid(density ~ num_qubits, labeller = labeller(density=label_both)) +
    theme_paper_base() +
    scale_x_continuous("s") +
    scale_y_continuous("Energy")

save_name <- str_c("energy")

pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = WIDTH, height = HEIGHT)
print(g)
dev.off()

g <- ggplot(d) +
    geom_line(aes(x = fraction, y = gap), colour = COLOURS.LIST[[3]]) +
    facet_grid(density ~ num_qubits, labeller = labeller(density=label_both)) +
    theme_paper_base() +
    scale_x_continuous("s") +
    scale_y_continuous("Spectral Gap")

save_name <- str_c("gap")

pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = WIDTH, height = HEIGHT)
print(g)
dev.off()

