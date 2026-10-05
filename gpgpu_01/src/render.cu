#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__device__ rgba8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgba8_t{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgba8_t{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgba8_t{r, 255, 0, 255};
  }
  else if (x < 1.0)
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgba8_t{255, b, 0, 255};
  }
  else
  {
    return rgba8_t{0, 0, 0, 255};
  }
}

/// Compute the number or iteration of the fractal per pixel and store the result in *buffer*.
/// Note that a 32-bits location can be used to store an integer (int32) or a color (uchar4).
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void compute_iter(char* buffer, int width, int height, size_t pitch, int max_iter) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height || x < 0 || y < 0)
    {
        return;
    }

    float w = (float)width / 3.5;
    float h = (float)height / 2;
    float mx0 = ((float)x/w) - 2.5;
    float my0 = ((float)y/h) - 1;
    float mx = 0.0;
    float my = 0.0;
    int iteration = 0;

    while(mx*mx + my*my < 4 && iteration < max_iter)
    {
        float mxtemp = mx*mx - my*my + mx0;
        my = 2*mx*my + my0;
        mx = mxtemp;
        iteration++;
    }

    uint32_t* lineptr = (uint32_t*)(buffer + y * pitch);
    lineptr[x] = iteration;
}

/// This function is single thread for now!
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
/// \param LUT Output look-up table
__global__ void compute_LUT(const char* buffer, int width, int height, size_t pitch, int max_iter, uchar4* LUT) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || blockIdx.y != 0 || threadIdx.y != 0)
    {
        return;
    }

    uint32_t* histo = (uint32_t*)LUT;
    for(int y = 0; y < height; y++)
    {
        uint32_t*  lineptr = (uint32_t*)(buffer + y * pitch);
        for(int x = 0; x < width; x++)
        {
            int k = lineptr[x];
            if (k <= max_iter)
                histo[k]++;
        }
    }

    int iteration = 0;
    uint32_t total = 0.0;
    for (int i = 0; i < max_iter; i++)
        total += histo[i];

    for (int k = 0; k <= max_iter; k++)
    {
        if (k == max_iter)
            LUT[k] = {0, 0, 0, 255};
        else
        {
            uint32_t count = histo[k];
            iteration += count;

            float x = total > 0 ? ((float)iteration / (float)total) : 0.0;
            rgba8_t color = heat_lut(x);
            LUT[k] = color;
        }
    }
}

///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void apply_LUT(char* buffer, int width, int height, size_t pitch, int max_iter, const uchar4* LUT) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height || x < 0 || y < 0)
    {
        return;
    }

    uint32_t*  lineptr = (uint32_t*)(buffer + y * pitch);
    int k = lineptr[x];

    uchar4 color = LUT[k];
    uchar4* lineptr_out = (uchar4*)(buffer + y * pitch);
    lineptr_out[x] = color;
}

// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch, int n_iterations) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float mx0 = -2.5f + ((float)x / (float)width) * 3.5f;
  float my0 = -1.0f + ((float)y / (float)height) * 2.0f;

  float mx = 0.0f;
  float my = 0.0f;
  int iteration = 0;

  while ((mx * mx + my * my < 4.0f) && (iteration < n_iterations)) {
    float mxtemp = mx * mx - my * my + mx0;
    my = 2.0f * mx * my + my0;
    mx = mxtemp;
    iteration++;
  }

  float grey = (float)iteration / (float)n_iterations;
  rgba8_t color = heat_lut(grey);
  rgba8_t* lineptr = (rgba8_t *)(buffer + y * pitch);
  lineptr[x] = color;
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations) {
  char* devBuffer = nullptr;
  size_t pitch = 0;

  cudaMallocPitch(&devBuffer, &pitch, width * sizeof(uchar4), height);

  int bsize = 32;
  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid((width + bsize - 1) / bsize, (height + bsize - 1) / bsize);

  mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations);
  cudaDeviceSynchronize();
  cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost);
  cudaFree(devBuffer);
}
