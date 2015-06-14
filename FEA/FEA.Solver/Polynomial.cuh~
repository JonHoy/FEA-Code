
// class for 3D polynomial

template <typename T, int SizeX, int SizeY = 1, int SizeZ = 1>
struct Polynomial {
	__device__ Polynomial() {
	
	}
	__device__ T Evaluate(T ValX, T ValY = 1, T ValZ = 1) {
		T sum = 0;
		int id = 0;
		for (int i = 0; i < SizeX; i++) {
			for (int j = 0; j < SizeY; j++) {
				for (int k = 0; k < SizeZ; k++)	{			
					T a = Coeffs[id];
					sum += a*Pow(ValX,i)*Pow(ValY,j)*Pow(ValZ,k);	
					id++; 
				}
			}
		}
		return sum;
	}
	__device__ T Differentiate(const int Dim, T ValX, T ValY = 1, T ValZ = 1) { // find the partial derivative at the specified point 
		T sum = 0;
		int id = 0;
		for (int i = 0; i < SizeX; i++) {
			for (int j = 0; j < SizeY; j++) {
				for (int k = 0; k < SizeZ; k++)	{			
					T a = Coeffs[id];
					if (Dim == 2) {
					    sum += k * a * Pow(ValZ,k-1) * Pow(ValX,i) * Pow(ValY, j);
					}
					else if (Dim == 1) {
					    sum += j * a * Pow(ValZ,k) * Pow(ValX,i) * Pow(ValY, j-1);
					}
					else // (Dim == 0 ) 	
					{
					    sum += i * a * Pow(ValZ,k) * Pow(ValX,i-1) * Pow(ValY, j);
					}
					id++;
				}
			}
		}
		return sum;
	}
	__device__ T& operator()(int i) {
		return Coeffs[i];
	}
	__device__ T& operator()(int i, int j) {
		return Coeffs[i * SizeY + j];
	}
	__device__ T& operator()(int i, int j, int k) {
		return Coeffs[i * SizeZ * SizeY + j * SizeZ + k];
	}
	T Coeffs[SizeX * SizeY * SizeZ];
	__device__ T Pow(T a, int b) {
		T ans = 1;
		for (int i = 0; i < b; i++) {
			ans *= a;
		}
		return ans;		
	}
};

