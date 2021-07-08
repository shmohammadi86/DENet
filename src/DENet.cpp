#include <RcppArmadillo.h>
#include <thread>

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

double r8_normal_01_cdf_inverse(double p);

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t thread_no,
                        Function fn) {
  if (thread_no <= 0) {
    thread_no = 1;
  }

  if (thread_no == 1) {
    for (size_t id = start; id < end; id++) {
      fn(id, 0);
    }
  } else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < thread_no; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);

          if ((id >= end)) {
            break;
          }

          try {
            fn(id, threadId);
          } catch (...) {
            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
    if (lastException) {
      std::rethrow_exception(lastException);
    }
  }
}


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

mat RIN_transform(mat A, int thread_no = 4) {
  int M = A.n_rows;
  int N = A.n_cols;

  mat Zr = zeros(M, N);
  ParallelFor(0, N, thread_no, [&](size_t i, size_t threadId) {
    vec v = A.col(i);

    uvec row_perm_forward = stable_sort_index(v);
    uvec row_perm = stable_sort_index(row_perm_forward);
    vec p = (row_perm + ones(size(row_perm))) / (row_perm.n_elem + 1);

    vec v_RINT = zeros(size(p));
    for (int j = 0; j < p.n_elem; j++) {
      double norm_inv = r8_normal_01_cdf_inverse(p(j));
      v_RINT(j) = norm_inv;
    }

    Zr.col(i) = v_RINT;
  });

  return (Zr);
}


// [[Rcpp::export]]
double F2z(double F, double d1, double d2) {
	double mu = d2 / (d2 - 2); // Only valud if d2 > 2
	double sigma_sq = (2*d2*d2*(d1+d2-2))/(d1*(d2-2)*(d2-2)*(d2-4)); // Only valid when d2 > 4
	
	double z = (F - mu) / sqrt(sigma_sq);	
	return(z);
}


// [[Rcpp::export]]
double T2z(double T, double nx, double ny, double p) {
	double m = (nx + ny - p - 1) / (p*(nx + ny - 2));	
	double z = F2z(T*m, p, nx + ny - p - 1);
	
	return(z);
}


// [[Rcpp::export]]
arma::sp_mat DENet(arma::sp_mat &G0, arma::mat &A, arma::uvec x, arma::uvec y, int normalization = 1) {
	double min_pval = 1e-300;

	double nx = x.n_elem;
	double ny = y.n_elem;
	
	if(nx < 3 | ny < 3) {
		fprintf(stderr, "Too few samples\n");
		return(NULL);
	}

	
	printf("Extracting submatrices (nx = %d, ny = %d)\n", (int)nx, (int)ny);
	// Make them 0-based
	x = x - 1;
	y = y - 1;

	mat Z = trans(A);
	switch(normalization) {
		case 1:
			Z = zscore(Z);
			break;
			
		case 2:
			Z = RIN_transform(Z);
			break;		
	}	
	
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
		idx(0) = it.row();
		idx(1) = it.col();
		
		arma::mat sub_S = S(idx, idx);
		arma::mat sub_Sinv = minv2x2(sub_S);		
		arma::vec sub_delta = delta(idx);
		
		mat sx = Sx(idx, idx);
		mat sy = Sy(idx, idx);
		
		double F = kappa1 * arma::mat(trans(sub_delta) * sub_Sinv * sub_delta)(0);
				
		(*it) = F2z(F, q, n - q - 1);

	}	
	G.replace(datum::nan, 0); 
	
	printf("Done\n");
	
	return(G);
}

// [[Rcpp::export]]
arma::mat DENet_full(arma::mat &A, arma::uvec x, arma::uvec y, int normalization = 1) {	
	double min_pval = 1e-300;
	
	
	double nx = x.n_elem;
	double ny = y.n_elem;
	
	if(nx < 3 | ny < 3) {
		fprintf(stderr, "Too few samples\n");
		return(NULL);
	}

	// Make them 0-based
	x = x - 1;
	y = y - 1;
	
	printf("Extracting submatrices (nx = %d, ny = %d)\n", (int)nx, (int)ny);
	mat Z = trans(A);
	switch(normalization) {
		case 1:
			Z = zscore(Z);
			break;
			
		case 2:
			Z = RIN_transform(Z);
			break;				
	}	
	
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
/*
	printf("\tPx variance ... \n");
	mat Ax_shifted = Ax.each_row() - mx;
	arma::mat Sx = trans(Ax_shifted) * Ax_shifted;
	
	printf("\tPy variance ... \n");
	mat Ay_shifted = Ay.each_row() - my;
	arma::mat Sy = trans(Ay_shifted) * Ay_shifted;

	printf("\tPooled variance ... \n");
	arma::mat S = ((nx - 1)*Sx + (ny - 1)*Sy) / (nx + ny - 2);	
	*/
	
	/*
	arma::mat Sx = cov(Ax);
	Sx.replace(datum::nan, 0); 

	arma::mat Sy = cov(Ay);	
	Sy.replace(datum::nan, 0); 
	
	arma::mat S = ((nx - 1)*Sx + (ny - 1)*Sy) / (nx + ny - 2);	
	
	rowvec mx = mean(Ax);
	rowvec my = mean(Ay);	
	arma::vec delta = trans(mx - my);
	printf("S: %dx%d, delta: %d\n", S.n_rows, S.n_cols, delta.n_elem);
	*/
	
	/*
	vec mu = trans(nx*mx + ny*my)/(nx+ny);
	vec sigma = S.diag();
	vec cv = mu / sigma;
	*/
	
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
			
			G(i, j) = G(j, i) = F2z(F, q, n - q - 1);
		}
	}			
	G.replace(datum::nan, 0); 
	printf("Done\n");

	return(G);
}


// [[Rcpp::export]]
arma::sp_mat kStarNN(arma::mat G, int sim2dist = 1, double LC = 1, int sym = 2) {
	int N = G.n_rows;
	
	mat dist(size(G));
	switch(sim2dist) {
		case 0:
			dist = G;
			break;
		
		case 1:
			dist = 1 / G;
			break;			

		case 2:
			double m = min(min(G));
			double M = max(max(G));
			dist = 1.0 - (G - m) / (M - m);
			break;
	}
	dist.replace(datum::nan, 0); 
	dist.diag().zeros();
	
	
	
	mat beta = LC * sort(dist, "ascend", 1);
	vec beta_sum = zeros(N);
	vec beta_sq_sum = zeros(N);


	mat lambda = zeros(size(dist));
	for (int k = 1; k < N; k++) {
		beta_sum += beta.col(k);
		beta_sq_sum += square(beta.col(k));

		lambda.col(k) = (1.0 / (double)k) *
						(beta_sum + sqrt(k + square(beta_sum) - k * beta_sq_sum));
	}
	lambda.replace(datum::nan, 0);
	lambda = trans(lambda);	
	beta = trans(beta);

	mat Delta = lambda - beta;
	Delta.shed_row(0);

	sp_mat G_puned(N, N);
	// for(int v = 0; v < N; v++) {
	ParallelFor(0, N, 1, [&](size_t v, size_t threadId) {
		vec delta = Delta.col(v);

		uvec rows = find(delta > 0, 1, "last");			
		int neighbor_no = rows.n_elem == 0? 0:rows(0);
		
		if(0 < neighbor_no) {
			vec w = delta(span(0, neighbor_no-1));
			w = w / sum(w);
			
			int dst = v;
			uvec v_idx = sort_index(trans(dist.row(v)), "ascend");
			for (int i = 0; i < neighbor_no; i++) {
			  int src = v_idx(i);
			  G_puned(src, dst) = w(i);
			}
		}
	});
	G_puned.replace(datum::nan, 0);  // replace each NaN with 0
	
	sp_mat G_sym;
	sp_mat Gt_pruned = trans(G_puned);			
	switch(sym) {
		case 0:
			G_sym = G_puned;
			break;
			
		case 1:		
			G_sym = (G_puned + Gt_pruned);
			G_sym.for_each([](sp_mat::elem_type &val) { val /= 2.0; });
			break;
			
		case 2:
			G_sym = sqrt(G_puned % Gt_pruned);
	}

	G_sym.diag().zeros();
	
	return(G_sym);
}
