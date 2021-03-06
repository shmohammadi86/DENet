# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

F2z <- function(F, d1, d2) {
    .Call(`_DENet_F2z`, F, d1, d2)
}

T2z <- function(T, nx, ny, p) {
    .Call(`_DENet_T2z`, T, nx, ny, p)
}

DENet <- function(G0, A, x, y, normalization = 1L) {
    .Call(`_DENet_DENet`, G0, A, x, y, normalization)
}

DENet_full <- function(A, x, y, normalization = 1L) {
    .Call(`_DENet_DENet_full`, A, x, y, normalization)
}

kStarNN <- function(G, sim2dist = 3L, LC = 1) {
    .Call(`_DENet_kStarNN`, G, sim2dist, LC)
}

symmetrize_network <- function(G, sym = 1L) {
    .Call(`_DENet_symmetrize_network`, G, sym)
}

