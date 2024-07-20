BASE.SIZE <- 25
HEIGHT = 14
WIDTH = 28
OUTDIR_PDF <- "img-pdf/"
COLOURS.LIST <- c("black", "#E69F00", "#999999", "#009371")

theme_paper_base <- function() {
    return(theme_bw(base_size=BASE.SIZE) +
           theme(axis.title.x = element_text(size = BASE.SIZE),
                 axis.title.y = element_text(size = BASE.SIZE),
                 legend.title = element_text(size = BASE.SIZE),
                 legend.position = "top",
                 plot.margin = unit(c(0,0,0,0), 'cm')))
}

create_save_locations <- function() {
    if (!dir.exists(OUTDIR_PDF)) {
        dir.create(OUTDIR_PDF, recursive = TRUE)
    }
}


