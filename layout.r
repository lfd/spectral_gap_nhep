BASE.SIZE <- 8
INCH.PER.CM <- 1/2.54
WIDTH <- 13.998*INCH.PER.CM
HEIGHT <- 7.28972*INCH.PER.CM*0.95
OUTDIR_PDF <- "img-pdf/"
OUTDIR_TIKZ <- "img-tikz/"
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


