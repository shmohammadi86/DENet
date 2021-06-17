#include "stats.hpp"

#define STATS_GO_INLINE
#define STATS_ENABLE_ARMA_WRAPPERS

double Fpval(double x, int df1, int df2) {
	return(1.0-stats::pf(x,df1,df2));
}
