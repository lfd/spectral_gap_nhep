#!/usr/bin/env Rscript

library(tidyverse)
library(ggh4x)
library(stringr)
library(tikzDevice)
library(patchwork)
library(rjson)
source("layout.r")

options(tikzLatexPackages = c(
    getOption("tikzLatexPackages"),
    "\\usepackage{amsmath}"
))

tikz <- FALSE
if (!tikz) {
    POINT.SIZE <- 0.2
    LINE.SIZE <- 0.5
}

create_save_locations(tikz)

param_path <- "proceedings_results/00_parameters/trackrecon_parameters.json"
qaoa_path <- "proceedings_results/05_qaoa/qaoa_track_reconstruction_results.csv"
as_path <- "proceedings_results/06_schedules/anneal_schedule_track_reconstruction_results.csv"
energy_path <- "proceedings_results/04_adiabatic/adiabatic_track_reconstruction_results.csv"

apply_version_to_df <- function(df, file) {
    df$version <- str_extract(
        file,
        "\\d{4}-\\d{2}-\\d{2}T\\d{2}\\.\\d{2}\\.\\d{2}\\.\\d{3}Z"
    )
    return(df)
}

read_csv_and_apply_version <- function(file) {
    df <- read.csv(file)
    return(apply_version_to_df(df, file))
}

read_json_and_apply_version <- function(file) {
    df <- fromJSON(file = file) %>% as.data.frame()
    return(apply_version_to_df(df, file))
}

load_data <- function(file_list, d_params) {
    df <- lapply(
        setNames(nm = file_list),
        read_csv_and_apply_version
    ) %>%
        bind_rows() %>%
        merge(d_params, by = "version")
    return(df)
}

param_file_list <- list.files(
    path = param_path,
    recursive = TRUE,
    full.names = TRUE,
)

qaoa_file_list <- list.files(
    path = qaoa_path,
    recursive = TRUE,
    full.names = TRUE,
)

as_file_list <- list.files(
    path = as_path,
    recursive = TRUE,
    full.names = TRUE,
)

energy_file_list <- list.files(
    path = energy_path,
    recursive = TRUE,
    full.names = TRUE,
)

d_params <- lapply(setNames(nm = param_file_list), read_json_and_apply_version) %>% bind_rows()
d_qaoa <- load_data(qaoa_file_list, d_params)
d_as <- load_data(as_file_list, d_params)
d_energy <- load_data(energy_file_list, d_params)
d_energy <- d_energy[!duplicated(d_energy[, setdiff(names(d_energy), "version")]), ]

qubit_labeller <- function(layer) {
    if (tikz) {
        paste0("\\# Qubits = ", layer)
    } else {
        paste0("# Qubits = ", layer)
    }
}

index_labeller <- function(layer) {
    paste0("i = ", layer)
}

q_labeller <- function(layer) {
    ifelse(layer == -1, "RI", ifelse(layer == "Optimum", "Optimum", paste0(ifelse(tikz, "$q_\\text{max} = $", "q = "), layer)))
}

d_energy <- d_energy %>%
    group_by(geometric_index, seed, num_angle_parts, data_fraction) %>%
    mutate(
        min_gap_frac = min(ifelse(gap == min(gap), fraction, 1)),
        min_gap = min(gap)
    ) %>%
    ungroup()

d_energy_selected <- d_energy %>% filter(geometric_index == 27 & data_fraction == 0.1)
d_energy_selected_msg <- d_energy_selected %>% filter(fraction == min_gap_frac)
d_energy_selected_msg$mid <- d_energy_selected_msg$gs + 0.5 * d_energy_selected_msg$gap

g <- ggplot(d_energy_selected, aes(x = fraction)) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = fes, colour = "First Excited State")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = fes, colour = "First Excited State")
    ) +
    geom_segment(
        aes(
            x = min_gap_frac,
            y = d_energy_selected_msg$fes,
            xend = min_gap_frac,
            yend = d_energy_selected_msg$gs,
        ),
        colour = COLOURS.LIST[[1]],
        linewidth = LINE.SIZE
    ) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = gs, colour = "Ground State")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = gs, colour = "Ground State")
    ) +
    theme_paper_base() +
    scale_x_continuous(
        ifelse(tikz, "$s$", "s"),
        breaks = seq(0, 1, by = 0.2), limits = c(0, 1)
    ) +
    scale_y_continuous("Energy") +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Ground State" = COLOURS.LIST[[4]],
            "First Excited State" = COLOURS.LIST[[3]]
        ),
        breaks = c(
            "Ground State",
            "First Excited State",
            "Minimum Spectral Gap"
        )
    )

g_inset <- ggplot(d_energy_selected, aes(x = fraction)) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = fes, colour = "First Excited State")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = fes, colour = "First Excited State")
    ) +
    geom_segment(
        aes(
            x = min_gap_frac,
            xend = min_gap_frac,
            y = d_energy_selected_msg$mid - 0.05,
            yend = d_energy_selected_msg$gs,
            colour = "Minimum Spectral Gap",
        ),
        lineend = "round",
        arrow = arrow(length = unit(0.1, "cm")),
        linewidth = LINE.SIZE
    ) +
    geom_segment(
        aes(
            x = min_gap_frac,
            xend = min_gap_frac,
            y = d_energy_selected_msg$mid + 0.05,
            yend = d_energy_selected_msg$fes,
            colour = "Minimum Spectral Gap",
        ),
        lineend = "round",
        arrow = arrow(length = unit(0.1, "cm")),
        linewidth = LINE.SIZE
    ) +
    geom_text(
        x = d_energy_selected_msg$min_gap_frac,
        y = d_energy_selected_msg$mid,
        size = 2,
        label = "d",
    ) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = gs, colour = "Ground State")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = gs, colour = "Ground State")
    ) +
    theme_paper_base() +
    scale_x_continuous(
        ifelse(tikz, "$s$", "s"),
        limits = c(0.8, 0.95),
        breaks = seq(0.0, 1.0, by = 0.05),
    ) +
    scale_y_continuous("Energy",
        limits = c(-2.1, -1.55),
        breaks = seq(-3, -1, by = 0.2),
    ) +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Ground State" = COLOURS.LIST[[4]],
            "First Excited State" = COLOURS.LIST[[3]]
        ),
        breaks = c(
            "Ground State",
            "First Excited State",
            "Minimum Spectral Gap"
        )
    ) +
    theme(
        legend.position = "none",
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
    )

g <- g + inset_element(g_inset, 0.45, 0.05, 0.95, 0.6)

save_name <- str_c("energy_selected")
create_plot(g, save_name, 0.5, 0.24, tikz)

g <- ggplot(d_energy_selected, aes(x = fraction, y = gap)) +
    geom_line(
        aes(colour = "Spectral Gap"),
        linewidth = LINE.SIZE,
    ) +
    geom_point(
        aes(colour = "Spectral Gap"),
        size = POINT.SIZE,
    ) +
    geom_vline(aes(
        xintercept = min_gap_frac,
        colour = "Minimum Spectral Gap"
    ), linewidth = LINE.SIZE) +
    theme_paper_base() +
    scale_x_continuous(
        ifelse(tikz, "$s$", "s"),
        breaks = seq(0, 1, by = 0.2), limits = c(0, 1)
    ) +
    scale_y_continuous("Spectral Gap", breaks = seq(0, 2, by = 0.5)) +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Spectral Gap" = COLOURS.LIST[[2]]
        ),
        breaks = c(
            "Spectral Gap",
            "Minimum Spectral Gap"
        )
    ) +
    scale_fill_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Spectral Gap" = COLOURS.LIST[[2]]
        ),
        breaks = c(
            "Spectral Gap",
            "Minimum Spectral Gap"
        )
    )

g_inset <- ggplot(d_energy_selected, aes(x = fraction, y = gap)) +
    geom_line(
        aes(colour = "Spectral Gap"),
        linewidth = LINE.SIZE,
    ) +
    geom_point(
        aes(colour = "Spectral Gap"),
        size = POINT.SIZE,
    ) +
    geom_vline(aes(
        xintercept = min_gap_frac,
        colour = "Minimum Spectral Gap"
    ), linewidth = LINE.SIZE) +
    theme_paper_base() +
    scale_x_continuous(
        ifelse(tikz, "$s$", "s"),
        limits = c(0.8, 0.95),
        breaks = seq(0.0, 1.0, by = 0.05),
    ) +
    scale_y_continuous("Energy",
        limits = c(0.32, 0.37),
        breaks = seq(0, 0.5, by = 0.02),
    ) +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Spectral Gap" = COLOURS.LIST[[2]]
        ),
        breaks = c(
            "Spectral Gap",
            "Minimum Spectral Gap"
        )
    ) +
    scale_fill_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Spectral Gap" = COLOURS.LIST[[2]]
        ),
        breaks = c(
            "Spectral Gap",
            "Minimum Spectral Gap"
        )
    ) +
    theme(
        legend.position = "none",
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
    )

g <- g + inset_element(g_inset, 0.4, 0.5, 0.95, 0.95)

save_name <- str_c("gap_selected")
create_plot(g, save_name, 0.5, 0.24, tikz)

d_energy$qubit_range <- ifelse(
    d_energy$num_qubits <= 7,
    "3-7",
    ifelse(d_energy$num_qubits <= 13, "8-13",
        ifelse(d_energy$num_qubits <= 18, "14-18", "19-23")
    )
)
d_energy$qubit_range <- factor(d_energy$qubit_range, levels = c("3-7", "8-13", "14-18", "19-23"))

c <- d_energy %>%
    group_by(qubit_range) %>%
    filter(fraction == 0) %>%
    count()
print(c)


d_energy_stats <- d_energy %>%
    group_by(qubit_range) %>%
    summarise(
        frac_min = min(min_gap_frac),
        frac_avg = mean(min_gap_frac),
        frac_max = max(min_gap_frac),
        frac_quartile1 = quantile(min_gap_frac, 0.25),
        frac_median = median(min_gap_frac),
        frac_quartile3 = quantile(min_gap_frac, 0.75),
        frac_sd = sd(min_gap_frac)
    )
d_energy_stats$frac_lower_bound <- d_energy_stats$frac_avg - d_energy_stats$frac_sd
d_energy_stats$frac_upper_bound <- d_energy_stats$frac_avg + d_energy_stats$frac_sd

d_energy_avgs <- d_energy %>%
    group_by(fraction, qubit_range) %>%
    summarise(
        gap_avg = mean(gap),
        gap_median = median(gap),
        gap_quartile1 = quantile(gap, 0.25),
        gap_quartile3 = quantile(gap, 0.75),
        gap_min = min(gap),
        gap_max = max(gap),
        gap_sd = sd(gap),
        fes_avg = mean(fes),
        fes_median = median(fes),
        fes_quartile1 = quantile(fes, 0.25),
        fes_quartile3 = quantile(fes, 0.75),
        fes_min = min(fes),
        fes_max = max(fes),
        fes_sd = sd(fes),
        gs_avg = mean(gs),
        gs_median = median(gs),
        gs_quartile1 = quantile(gs, 0.25),
        gs_quartile3 = quantile(gs, 0.75),
        gs_min = min(gs),
        gs_max = max(gs),
        gs_sd = sd(gs),
    ) %>%
    merge(d_energy_stats)

d_energy_avgs$gap_lower_bound <- d_energy_avgs$gap_avg - d_energy_avgs$gap_sd
d_energy_avgs$gap_upper_bound <- d_energy_avgs$gap_avg + d_energy_avgs$gap_sd
d_energy_avgs$fes_lower_bound <- d_energy_avgs$fes_avg - d_energy_avgs$fes_sd
d_energy_avgs$fes_upper_bound <- d_energy_avgs$fes_avg + d_energy_avgs$fes_sd
d_energy_avgs$gs_lower_bound <- d_energy_avgs$gs_avg - d_energy_avgs$gs_sd
d_energy_avgs$gs_upper_bound <- d_energy_avgs$gs_avg + d_energy_avgs$gs_sd

g <- ggplot(d_energy_avgs, mapping = aes(x = fraction)) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = gap_avg, colour = "Spectral Gap")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = gap_avg, colour = "Spectral Gap")
    ) +
    geom_vline(aes(
        xintercept = frac_avg,
        colour = "Minimum Spectral Gap"
    ), linewidth = LINE.SIZE) +
    geom_ribbon(
        aes(
            y = 0,
            xmin = frac_lower_bound, xmax = frac_upper_bound, fill = "Minimum Spectral Gap"
        ),
        alpha = 0.4,
    ) +
    geom_rect(
        aes(
            ymin = -Inf, ymax = Inf,
            xmin = frac_lower_bound, xmax = frac_upper_bound,
            fill = "Minimum Spectral Gap",
        ),
        colour = NA,
        alpha = 0.006,
    ) +
    geom_ribbon(
        mapping = aes(
            ymin = gap_lower_bound,
            ymax = gap_upper_bound, fill = "Spectral Gap"
        ),
        alpha = 0.4,
    ) +
    theme_paper_base() +
    scale_x_continuous(
        ifelse(tikz, "$s$", "s"),
        breaks = seq(0, 1, by = 0.2), limits = c(0, 1.1)
    ) +
    scale_y_continuous("Spectral Gap") +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Spectral Gap" = COLOURS.LIST[[2]]
        ),
        breaks = c(
            "Spectral Gap",
            "Minimum Spectral Gap"
        )
    ) +
    scale_fill_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Spectral Gap" = COLOURS.LIST[[2]]
        ),
        breaks = c(
            "Spectral Gap",
            "Minimum Spectral Gap"
        )
    ) +
    facet_wrap(qubit_range ~ .,
        labeller = labeller(
            qubit_range = qubit_labeller
        ),
    ) +
    theme(legend.position = "right")

save_name <- str_c("gap_stat")
create_plot(g, save_name, 1, 0.22, tikz)

g <- ggplot(d_energy_avgs, mapping = aes(x = fraction)) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = fes_avg, colour = "First Excited State")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = fes_avg, colour = "First Excited State")
    ) +
    geom_vline(
        aes(
            xintercept = frac_avg,
            colour = "Minimum Spectral Gap",
        ),
        linewidth = LINE.SIZE
    ) +
    geom_ribbon(
        aes(
            y = 0,
            xmin = frac_lower_bound, xmax = frac_upper_bound,
            fill = "Minimum Spectral Gap"
        ),
        alpha = 0.4,
    ) +
    geom_rect(
        aes(
            ymin = -Inf, ymax = Inf,
            xmin = frac_lower_bound, xmax = frac_upper_bound,
            fill = "Minimum Spectral Gap",
        ),
        colour = NA,
        alpha = 0.006,
    ) +
    geom_ribbon(
        mapping = aes(
            ymin = fes_lower_bound,
            ymax = fes_upper_bound, fill = "First Excited State"
        ),
        alpha = 0.2,
    ) +
    geom_line(
        linewidth = LINE.SIZE,
        mapping = aes(y = gs_avg, colour = "Ground State")
    ) +
    geom_point(
        size = POINT.SIZE,
        mapping = aes(y = gs_avg, colour = "Ground State")
    ) +
    geom_ribbon(
        mapping = aes(
            ymin = gs_lower_bound,
            ymax = gs_upper_bound, fill = "Ground State"
        ),
        alpha = 0.2,
    ) +
    theme_paper_base() +
    scale_x_continuous(
        ifelse(tikz, "$s$", "s"),
        breaks = seq(0, 1, by = 0.2), limits = c(0, 1.1)
    ) +
    scale_y_continuous("Energy") +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Ground State" = COLOURS.LIST[[4]],
            "First Excited State" = COLOURS.LIST[[3]]
        ),
        breaks = c(
            "Ground State",
            "First Excited State",
            "Minimum Spectral Gap"
        )
    ) +
    scale_fill_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Ground State" = COLOURS.LIST[[4]],
            "First Excited State" = COLOURS.LIST[[3]]
        ),
        breaks = c(
            "Ground State",
            "First Excited State",
            "Minimum Spectral Gap"
        )
    ) +
    facet_wrap(qubit_range ~ .,
        labeller = labeller(
            qubit_range = qubit_labeller
        ),
    ) +
    theme(legend.position = "right")

save_name <- str_c("energy_stat")
create_plot(g, save_name, 1, 0.2, tikz)

d_qaoa$q <- as.factor(d_qaoa$q)
d_qaoa$appr_ratio <- d_qaoa$qaoa_energy / d_qaoa$min_energy
d_qaoa$appr_ratio[d_qaoa$appr_ratio < 0] <- 0

g <- ggplot(d_qaoa) +
    geom_point(aes(x = p, y = appr_ratio, colour = q),
        size = POINT.SIZE
    ) +
    geom_line(aes(x = p, y = appr_ratio, colour = q),
        linewidth = LINE.SIZE
    ) +
    geom_hline(aes(yintercept = 1.0, colour = "Optimum"), linetype = "dashed") +
    theme_paper_base() +
    scale_x_continuous(ifelse(tikz, "$p$", "p"), limits=c(1, 50)) +
    scale_y_continuous("Appr. Ratio", breaks = seq(-2, 1, by = 0.2)) +
    scale_color_manual("", values = COLOURS.LIST, labels = q_labeller) +
    theme(
        legend.position = "right",
        legend.text = element_text(margin = margin(l = 1, unit = "mm")),
        legend.key.height = unit(4, "mm")
    )

save_name <- str_c("qaoa_energy")

create_plot(g, save_name, 0.5, 0.15, tikz)

d_as <- d_as %>%
    group_by(geometric_index, q, optimiser) %>%
    mutate(rel_anneal_time = anneal_time / max(anneal_time)) %>%
    ungroup() %>%
    merge(d_energy_selected, by = c("geometric_index"), all.x = TRUE)
d_as$q <- as.factor(d_as$q.x)
d_as$optimiser <- d_as$optimiser.x

g <- ggplot(d_as) +
    geom_point(aes(x = rel_anneal_time, y = anneal_fraction, colour = "Anneal Fraction"),
        size = POINT.SIZE
    ) +
    geom_line(aes(x = rel_anneal_time, y = anneal_fraction, colour = "Anneal Fraction"),
        linewidth = LINE.SIZE
    ) +
    geom_hline(aes(yintercept = min_gap_frac, colour = "Minimum Spectral Gap")) +
    facet_wrap(. ~ q,
        labeller = labeller(
            q = q_labeller
        ),
    ) +
    theme_paper_base() +
    scale_x_continuous("Anneal Time Fraction", breaks = seq(0, 1, by = 0.5)) +
    scale_y_continuous(ifelse(tikz, "$s$", "s")) +
    scale_colour_manual("",
        values = c(
            "Minimum Spectral Gap" = COLOURS.LIST[[1]],
            "Anneal Fraction" = COLOURS.LIST[[5]]
        ),
        breaks = c(
            "Anneal Fraction",
            "Minimum Spectral Gap"
        )
    )

save_name <- str_c("anneal_schedule")
create_plot(g, save_name, 0.5, 0.25, tikz)

# only use last p
d_qaoa_reshaped <- d_qaoa %>% filter(p == max_p)

d_qaoa_reshaped <- d_qaoa_reshaped %>%
    # First, pivot the beta columns to long format
    pivot_longer(
        cols = matches("beta|gamma"),
        names_to = c(".value", "index"),
        names_pattern = "(beta|gamma)(\\d+)",
    ) %>%
    mutate(index = as.numeric(index))

d_qaoa_reshaped$beta <- abs(d_qaoa_reshaped$beta)
d_qaoa_reshaped$gamma <- abs(d_qaoa_reshaped$gamma)

g <- ggplot(d_qaoa_reshaped) +
    geom_point(aes(
        x = index, y = gamma,
        colour = "gamma"
    ), size = POINT.SIZE) +
    geom_line(aes(
        x = index, y = gamma,
        colour = "gamma"
    ), linewidth = LINE.SIZE) +
    geom_point(aes(
        x = index, y = beta,
        colour = "beta"
    ), size = POINT.SIZE) +
    geom_line(aes(
        x = index, y = beta,
        colour = "beta"
    ), linewidth = LINE.SIZE) +
    theme_paper_base() +
    scale_colour_manual(
        "",
        values = c(
            "beta" = COLOURS.LIST[[4]],
            "gamma" = COLOURS.LIST[[2]]
        ),
        labels = c(
            ifelse(tikz, "$\\beta$", "beta"),
            ifelse(tikz, "$\\gamma$", "gamma")
        )
    ) +
    facet_wrap(. ~ q,
        labeller = labeller(q = q_labeller)
    ) +
    scale_x_continuous(
        ifelse(tikz, "$p$", "p"),
        breaks = seq(0, 50, by = 10)
    ) +
    scale_y_continuous("Absolute Parameter Value")


save_name <- str_c("qaoa_params")
create_plot(g, save_name, 0.5, 0.25, tikz)
