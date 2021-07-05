#include <RcppArmadillo.h>

using namespace arma;
using namespace std;


#ifdef _OPENMP
#include <omp.h>
#endif

#include <DENet.h>

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

#define ARMA_USE_CXX11_RNG
#define DYNSCHED

arma::mat minv2x2(arma::mat A) {
	if(A.n_rows != 2 | A.n_cols != 2) {
		fprintf(stderr, "minv2x2: Matrix should be 2x2. return Null\n");
	}
	arma::mat Ainv = zeros(size(A));
	Ainv(0, 0) = A(1, 1);
	Ainv(0, 1) = -A(0, 1);
	Ainv(1, 0) = -A(1, 0);
	Ainv(1, 1) = A(0, 0);
	
	double denom = A(0, 0)*A(1, 1) - A(0, 1)*A(1, 0);
	Ainv /= denom;
	
	return(Ainv);
}

mat zscore(mat A) {
  rowvec mu = mean(A, 0);
  rowvec sigma = stddev(A, 0);

  for (int j = 0; j < A.n_cols; j++) {
    A.col(j) = (A.col(j) - mu(j)) / sigma(j);
  }

  return A;
}

// [[Rcpp::export]]
arma::field<arma::sp_mat> DENet(arma::sp_mat &G0, arma::mat &A, arma::uvec x, arma::uvec y, bool logpval_weights = false, double min_pval = 1e-300) {
	arma::field<arma::sp_mat> out(2);

	double nx = x.n_elem;
	double ny = y.n_elem;
	
	if(nx < 3 | ny < 3) {
		fprintf(stderr, "Too few samples\n");
		return(out);
	}

	
	printf("Extracting submatrices (nx = %d, ny = %d)\n", (int)nx, (int)ny);
	// Make them 0-based
	x = x - 1;
	y = y - 1;

	mat Z = zscore(trans(A));
	
	arma::mat Ax = Z.rows(x);
	arma::mat Ay = Z.rows(y);

	printf("Computing pairwise statistics ... ");
	arma::mat Sx = cov(Ax);
	Sx.replace(datum::nan, 0); 

	arma::mat Sy = cov(Ay);	
	Sy.replace(datum::nan, 0); 
	
	arma::mat S = ((nx - 1)*Sx + (ny - 1)*Sy) / (nx + ny - 2);	
	
	rowvec mx = mean(Ax);
	rowvec my = mean(Ay);	
	arma::vec delta = trans(mx - my);
	
	vec mu = trans(nx*mx + ny*my)/(nx+ny);
	vec sigma = S.diag();
	vec cv = mu / sigma;
		
	// Update the network
	printf("Updating edge weights\n");
	
	arma::sp_mat G = G0;	
	arma::sp_mat::iterator it     = G.begin();
	arma::sp_mat::iterator it_end = G.end();

	arma::uvec idx(2);
	double n = nx + ny, q = 2;
	double kappa1 = (nx*ny)/(n);
	double kappa2 = (n - q - 1) / (q*(n - 2));
	double kappa = kappa1 * kappa2;
	for(; it != it_end; ++it) {
		double w = logpval_weights? pow(10, (*it)):(*it);
		
		idx(0) = it.row();
		idx(1) = it.col();
		
		arma::mat sub_S = S(idx, idx);
		arma::mat sub_Sinv = minv2x2(sub_S);		
		arma::vec sub_delta = delta(idx);
		
		mat sx = Sx(idx, idx);
		mat sy = Sy(idx, idx);
		
		double F = kappa1 * arma::mat(trans(sub_delta) * sub_Sinv * sub_delta)(0);
				
		(*it) = F;
	}	
	G.replace(datum::nan, 0); 
	
	
	arma::sp_mat Gp = G;	
	for(it = Gp.begin(); it != Gp.end(); ++it) {
		double F = (*it);
		
		int df1 = q;
		int df2 = n - q - 1;
		double pval = Fpval(F,df1,df2);

		//pval = logpval_weights? (2*w*pval/(w+pval)):pval; // Combine p-values, if needed, using Harmonic mean of p-values (https://www.pnas.org/content/116/4/1195)	
		pval = max(pval, min_pval);
		
		(*it) = -log10(pval);
	}	

	printf("Done\n");
	
	out(0) = G;
	out(1) = Gp;
	
		
	return(out);
}

// [[Rcpp::export]]
arma::field<arma::mat> DENet_full(arma::mat &A, arma::uvec x, arma::uvec y, double min_pval = 1e-300) {	
	arma::field<arma::mat> out(2);
	
	double nx = x.n_elem;
	double ny = y.n_elem;
	
	if(nx < 3 | ny < 3) {
		fprintf(stderr, "Too few samples\n");
		return(out);
	}

	// Make them 0-based
	x = x - 1;
	y = y - 1;
	
	printf("Extracting submatrices (nx = %d, ny = %d)\n", (int)nx, (int)ny);
	mat Z = zscore(trans(A));
	
	
	arma::mat Ax = Z.rows(x);
	arma::mat Ay = Z.rows(y);
	
	printf("Computing pairwise statistics ... ");
	arma::mat Sx = cov(Ax);
	Sx.replace(datum::nan, 0); 

	arma::mat Sy = cov(Ay);	
	Sy.replace(datum::nan, 0); 
	
	arma::mat S = ((nx - 1)*Sx + (ny - 1)*Sy) / (nx + ny - 2);	
	
	rowvec mx = mean(Ax);
	rowvec my = mean(Ay);	
	arma::vec delta = trans(mx - my);
	printf("S: %dx%d, delta: %d\n", S.n_rows, S.n_cols, delta.n_elem);
	
	vec mu = trans(nx*mx + ny*my)/(nx+ny);
	vec sigma = S.diag();
	vec cv = mu / sigma;
	// Update the network
	printf("Updating edge weights\n");
	
	arma::mat G = zeros(size(S));
	arma::mat Gp = zeros(size(S));

	arma::uvec idx(2);
	double n = nx + ny, q = 2;
	double kappa1 = (nx*ny)/(n);
	double kappa2 = (n - q - 1) / (q*(n - 2));
	double kappa = kappa1 * kappa2;
	
	for(int i = 0; i < S.n_rows; i++) {
		for(int j = i+1; j < S.n_cols; j++) {							
			idx(0) = i;
			idx(1) = j;
			
			arma::mat sub_S = S(idx, idx);
			arma::mat sub_Sinv = minv2x2(sub_S);		
			arma::vec sub_delta = delta(idx);
			
			double F = kappa1 * arma::mat(trans(sub_delta) * sub_Sinv * sub_delta)(0);
						
			int df1 = q;
			int df2 = n - q - 1;
			double pval = Fpval(F,df1,df2);
			pval = max(pval, min_pval);
			
			G(i, j) = G(j, i) = F;
			Gp(i, j) = Gp(j, i) = -log10(pval);
		}
	}			
	G.replace(datum::nan, 0); 
	Gp.replace(datum::nan, 0); 

	printf("Done\n");
	
	out(0) = G;
	out(1) = Gp;
	
	return(out);
}
