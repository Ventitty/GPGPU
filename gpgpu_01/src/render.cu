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

rgba8_t heat_lut(float x)
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

// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch, uchar4* LUT)
{
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

  uint8_t grayv = (uint8_t)((255 * iteration) / n_iterations);
  uchar4* lineptr = (uchar4*)(buffer + y * pitch);
  lineptr[x] = make_uchar4(grayv, grayv, grayv, 255);
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
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
