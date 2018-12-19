#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>

#define B 64
#define threadNum 32

const int INF = 1000000000;
void input(char *inFileName);
void output(char *outFileName);

void block_FW();
int ceil(int a, int b);
__global__ void phase1(int* dist, int Round, int n, size_t pitch);
__global__ void phase2(int* dist, int Round, int start_block, int end_block, int n, size_t pitch);
__global__ void phase3(int* dist, int Round, int start_block, int end_block, int n, size_t pitch);

int n, m, rank, size;	
int *Dist = NULL;
int *device_Dist = NULL;
size_t pitch;

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	input(argv[1]);
	block_FW();
	if(rank==0){
		output(argv[2]);
	}
	cudaFreeHost(Dist);
	MPI_Finalize();	
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
	fread(&m, sizeof(int), 1, file);

	cudaMallocHost(&Dist, (size_t)n*n*sizeof(int));

    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
			Dist[i*n+j] = (i==j) ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName) {
	FILE *outfile = fopen(outFileName, "wb");
	fwrite(Dist, sizeof(int), n*n, outfile);	
    fclose(outfile);
}

int ceil(int a, int b) {
	return (a + b - 1) / b;
}

void block_FW() {
	unsigned int round = ceil(n, B);
	dim3 block_p1 = {1, 1};
	dim3 block_p2 = {2, round};
	dim3 block_p3 = {round, round};
	dim3 threads = {threadNum, threadNum};
	int start_block[2] = {0, (int)round/2}, end_block[2] = {(int)round/2, (int)round}, offset[2]={((int)round/2)*B, n-((int)round/2)*B};
	MPI_Request req1, req2;
	cudaMallocPitch(&device_Dist, &pitch, (size_t)n*sizeof(int), (size_t)n);
	cudaMemcpy2D((int*)((char*)device_Dist+start_block[rank]*B*pitch), pitch, Dist+start_block[rank]*B*n, (size_t)n*sizeof(int), (size_t)n*sizeof(int), (size_t)offset[rank], cudaMemcpyHostToDevice);
	int *swap_region;
	cudaMallocHost(&swap_region, (size_t)B*B*sizeof(int));
	for (unsigned int r = 0; r < round; ++r) {
		int cur = (r>=start_block[rank] && r<end_block[rank]) ? 1 : 0;
		int height = (r<round-1) ? B : n-r*B, width = height;
		if(cur){
			phase1<<<block_p1, threads>>>(device_Dist, r, n, pitch);
			cudaMemcpy2D(swap_region, (size_t)B*sizeof(int), (int*)((char*)device_Dist+r*B*pitch)+r*B, pitch, (size_t)width*sizeof(int), (size_t)height, cudaMemcpyDeviceToHost);
			MPI_Isend(swap_region, B*B, MPI_INT, 1-rank, r, MPI_COMM_WORLD, &req1);	
		}
		else{
			MPI_Recv(swap_region, B*B, MPI_INT, 1-rank, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			cudaMemcpy2D((int*)((char*)device_Dist+r*B*pitch)+r*B, pitch, swap_region, (size_t)B*sizeof(int),  (size_t)width*sizeof(int), (size_t)height, cudaMemcpyHostToDevice);
		}
		phase2<<<block_p2, threads>>>(device_Dist, r, start_block[rank], end_block[rank], n, pitch);
		if(cur){
			cudaMemcpy2D(Dist+r*B*n, (size_t)n*sizeof(int), (int*)((char*)device_Dist+r*B*pitch), pitch, (size_t)n*sizeof(int), (size_t)height, cudaMemcpyDeviceToHost);	
			MPI_Isend(Dist+r*B*n, n*height, MPI_INT, 1-rank, round+r, MPI_COMM_WORLD, &req2);	
		}else{
			MPI_Recv(Dist+r*B*n, n*height, MPI_INT, 1-rank, round+r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
			cudaMemcpy2D((int*)((char*)device_Dist+r*B*pitch), pitch, Dist+r*B*n, (size_t)n*sizeof(int), (size_t)n*sizeof(int), (size_t)height, cudaMemcpyHostToDevice);
		}
		phase3<<<block_p3, threads>>>(device_Dist, r, start_block[rank], end_block[rank], n, pitch);	
		if(cur){
			MPI_Wait(&req1, MPI_STATUS_IGNORE);
			MPI_Wait(&req2, MPI_STATUS_IGNORE);
		}
	}
	cudaMemcpy2D(Dist+start_block[rank]*B*n, (size_t)n*sizeof(int), (int*)((char*)device_Dist+start_block[rank]*B*pitch), pitch, (size_t)n*sizeof(int), (size_t)offset[rank], cudaMemcpyDeviceToHost);
	MPI_Bcast(Dist+start_block[1]*B*n, n*offset[1], MPI_INT, 1, MPI_COMM_WORLD);
	cudaFree(device_Dist);
	cudaFreeHost(swap_region);
}

__global__ void phase1(int* dist, int Round, int n, size_t pitch){

	int base = Round*B;
	int i_st = base + threadIdx.x*(B/threadNum), i_ed = i_st + (B/threadNum);
	int j_st = base + threadIdx.y*(B/threadNum), j_ed = j_st + (B/threadNum);

	if(i_ed > n){
		i_ed = n;
	}
	if(j_ed > n){
		j_ed = n;
	}

	__shared__ int sm[B][B];

	#pragma unroll
	for(int i=i_st ; i<i_ed ; ++i){
		#pragma unroll
		for(int j=j_st ; j<j_ed ; ++j){
			int *dij = (int*)((char*)dist+pitch*i)+j;
			sm[i-base][j-base] = *dij;
		}
	}
	__syncthreads();

	int len = ((Round+1)*B < n) ? B : n - (Round)*B;

	#pragma unroll
	for (int k = 0; k < len; ++k) {
		#pragma unroll
		for(int i = i_st; i<i_ed ; ++i){
			#pragma unroll
			for(int j = j_st ; j<j_ed ; ++j){
				int relax = sm[i-base][k] + sm[k][j-base];
				if(relax < sm[i-base][j-base]){
					sm[i-base][j-base] = relax;
				}
			}
		}
		__syncthreads();
	}

	#pragma unroll
	for(int i=i_st ; i<i_ed ; ++i){
		#pragma unroll
		for(int j=j_st ; j<j_ed ; ++j){
			int *dij = (int*)((char*)dist+pitch*i)+j;
			*dij = sm[i-base][j-base];
		}
	}
}

__global__ void phase2(int* dist, int Round, int start_block, int end_block, int n, size_t pitch){
	if(blockIdx.y==Round)
		return;

	__shared__ int sm[2][B][B];
	
	int base_i = (1-blockIdx.x)*Round*B + blockIdx.x*blockIdx.y*B;
	int base_j = blockIdx.x*Round*B + (1-blockIdx.x)*blockIdx.y*B;
	int shift = B/threadNum;
	int i_st = base_i + threadIdx.x*shift, i_ed = i_st + shift; 
	int j_st = base_j + threadIdx.y*shift, j_ed = j_st + shift;	

	if(i_ed < B*start_block || i_st >= B*end_block)
		return;
	
	#pragma unroll
	for(int i=i_st ; i<i_ed ; ++i){
		#pragma unroll
		for(int j=j_st ; j<j_ed ; ++j){
			if(i<n && j<n){
				int *dij = (int*)((char*)dist+pitch*i)+j;
				sm[0][i-base_i][j-base_j] = *dij;
			}
			if(Round*B+(i-base_i)<n && Round*B+(j-base_j)<n){
				int *dkk = (int*)((char*)dist+pitch*(Round*B+(i-base_i))) + Round*B+(j-base_j);
				sm[1][i-base_i][j-base_j] = *dkk;
			}
		}
	}
	__syncthreads();

	if(i_ed > n){
		i_ed = n;
	}
	if(j_ed > n){
		j_ed = n;
	}
	int len = ((Round+1)*B < n) ? B : n - (Round)*B;
	int i_offset = i_st-base_i, i_len = i_ed - i_st;
	int j_offset = j_st-base_j, j_len = j_ed - j_st;
	
	#pragma unroll
	for(int i=i_offset ; i<i_offset+i_len ; ++i){
		#pragma unroll
		for(int j=j_offset ; j<j_offset+j_len ; ++j){
			#pragma unroll
			for (int k = 0; k < len; ++k) {
				int relax = sm[1-blockIdx.x][i][k] + sm[blockIdx.x][k][j];
				if(relax < sm[0][i][j]){
					sm[0][i][j] = relax;
				}
			}
			int *dij = (int*)((char*)dist+pitch*(base_i+i))+base_j+j;
			*dij = sm[0][i][j];
		}
	}
}

__global__ void phase3(int* dist, int Round, int start_block, int end_block, int n, size_t pitch){
	if(blockIdx.x==Round || blockIdx.y==Round)
		return;

	if(blockIdx.x < start_block || blockIdx.x >= end_block)
		return;

	__shared__ int sm[2][B][B];
	
	int base_i = blockIdx.x*B;
	int base_j = blockIdx.y*B;
	int shift = B/threadNum;
	int i_st = base_i + threadIdx.x*shift, i_ed = i_st + shift;
	int j_st = base_j + threadIdx.y*shift, j_ed = j_st + shift;
	
	#pragma unroll
	for(int i=i_st ; i<i_ed ; ++i){
		#pragma unroll
		for(int j=j_st ; j<j_ed ; ++j){
			if(i<n && Round*B+(j-base_j)<n){
				int *dik = (int*)((char*)dist+pitch*i)+Round*B+(j-base_j);
				sm[0][j-base_j][i-base_i] = *dik;
			}
			if(Round*B+(i-base_i)<n && j<n){
				int *dkj = (int*)((char*)dist+pitch*(Round*B+(i-base_i)))+j;
				sm[1][i-base_i][j-base_j] = *dkj;
			}
		}
	}
	__syncthreads();
	
	if(i_ed > n){
		i_ed = n;
	}
	if(j_ed > n){
		j_ed = n;
	}
	int len = ((Round+1)*B < n) ? B : n - (Round)*B;
	int i_offset = i_st-base_i, i_len = i_ed - i_st;
	int j_offset = j_st-base_j, j_len = j_ed - j_st;
	
	#pragma unroll
	for(int i = 0 ; i < i_len ; ++i){
		#pragma unroll
		for(int j= 0 ; j < j_len ; ++j){
			int *dij = (int*)((char*)dist+pitch*(i_st+i))+j_st+j;
			int ans = *dij;
			#pragma unroll
			for (int k = 0; k < len; ++k) {
				int relax = sm[0][k][i_offset+i] + sm[1][k][j_offset+j];
				if(relax < ans){
					ans = relax;
				}
			}
			*dij = ans;
		}
	}
}