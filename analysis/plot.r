#!/usr/bin/env Rscript

library(tidyverse)
library(ggh4x)
library(stringr)
library(tikzDevice)
source("layout.r")

create_save_locations()

d <- read.csv("results/TR/spectral_gap_evolution.csv",
              stringsAsFactors = FALSE)
d$fraction <- as.numeric(d$fraction)
d$gs <- as.numeric(d$gs)
d$fes <- as.numeric(d$fes)
d$gap <- as.numeric(d$gap)
d$num_qubits <- as.numeric(d$num_qubits)

qubit_labeller <- function(layer) {
    paste0("\\# Qubits = ", layer)
}

index_labeller <- function(layer) {
    paste0("i = ", layer)
}


g <- ggplot(d) +
    geom_line(aes(x = fraction, y = gs), colour = COLOURS.LIST[[1]]) +
    geom_line(aes(x = fraction, y = fes), colour = COLOURS.LIST[[2]]) +
    facet_wrap(geometric_index ~ num_qubits, labeller = labeller(geometric_index=index_labeller, num_qubits=qubit_labeller)) +
    theme_paper_base() +
    scale_x_continuous("s") +
    scale_y_continuous("Energy")

save_name <- str_c("energy")

pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = WIDTH, height = HEIGHT*2)
print(g)
dev.off()
#tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = WIDTH, height = HEIGHT)
#print(g)
#dev.off()

g <- ggplot(d) +
    geom_line(aes(x = fraction, y = gap), colour = COLOURS.LIST[[3]]) +
    facet_wrap(geometric_index ~ num_qubits, labeller = labeller(geometric_index=index_labeller, num_qubits=qubit_labeller)) +
    theme_paper_base() +
    scale_x_continuous("s", breaks=seq(0, 1, by = 0.5)) +
    scale_y_continuous("Spectral Gap", breaks =seq(0, 2, by=0.5)) +
    labs(title="TR")

save_name <- str_c("gap")

pdf(str_c(OUTDIR_PDF, save_name, ".pdf"), width = WIDTH, height = HEIGHT*2)
print(g)
dev.off()
#tikz(str_c(OUTDIR_TIKZ, save_name, ".tex"), width = WIDTH, height = HEIGHT*2)
#print(g)
#dev.off()
