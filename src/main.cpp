#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <sstream>

#include <cuda_runtime.h>
#include <helper_cuda.h> // CUDA helper functions

// Display CUDA and NPP version info
bool printNPPInfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version: %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version:  %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    return checkCudaCapabilities(1, 0); // Minimum: SM 1.0
}

// Helper function to split string by delimiter
std::vector<std::string> splitString(const std::string &input, char delimiter)
{
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter))
        result.push_back(token);
    return result;
}

// Template for all filters to reduce repetition
#define FILTER_START                                                         \
    std::cout << "Processing: " << filePath << std::endl;                    \
    npp::ImageCPU_8u_C1 hostSrc;                                             \
    npp::loadImage(filePath, hostSrc);                                       \
    npp::ImageNPP_8u_C1 deviceSrc(hostSrc);                                  \
    const NppiSize size = {(int)deviceSrc.width(), (int)deviceSrc.height()}; \
    const NppiPoint offset = {0, 0};                                         \
    npp::ImageNPP_8u_C1 deviceDst(size.width, size.height);

#define FILTER_END                                        \
    npp::ImageCPU_8u_C1 hostDst(deviceDst.size());        \
    deviceDst.copyTo(hostDst.data(), hostDst.pitch());    \
    saveImage(outputFile, hostDst);                       \
    std::cout << "Saved to: " << outputFile << std::endl; \
    nppiFree(deviceSrc.data());                           \
    nppiFree(deviceDst.data());                           \
    nppiFree(hostSrc.data());                             \
    nppiFree(hostDst.data());

// Individual Filters

void cannyFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        FILTER_START

        const Npp16s lowThreshold = 72;
        const Npp16s highThreshold = 256;
        int bufferSize = 0;
        Npp8u *scratch = nullptr;

        NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(size, &bufferSize));
        cudaMalloc((void **)&scratch, bufferSize);

        NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), size, offset,
            deviceDst.data(), deviceDst.pitch(), size,
            NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
            lowThreshold, highThreshold, nppiNormL2,
            NPP_BORDER_REPLICATE, scratch));

        cudaFree(scratch);
        FILTER_END
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "Canny Filter Error: " << e << "\nAborting.\n";
        exit(EXIT_FAILURE);
    }
}

void sobelFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        FILTER_START
        NPP_CHECK_NPP(nppiFilterSobelHorizBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), size, offset,
            deviceDst.data(), deviceDst.pitch(), size,
            NPP_BORDER_REPLICATE));
        FILTER_END
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "Sobel Filter Error: " << e << "\nAborting.\n";
        exit(EXIT_FAILURE);
    }
}

void gaussFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        FILTER_START
        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), size, offset,
            deviceDst.data(), deviceDst.pitch(), size,
            NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE));
        FILTER_END
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "Gaussian Filter Error: " << e << "\nAborting.\n";
        exit(EXIT_FAILURE);
    }
}

void sharpenFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        FILTER_START
        NPP_CHECK_NPP(nppiFilterSharpenBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), size, offset,
            deviceDst.data(), deviceDst.pitch(), size,
            NPP_BORDER_REPLICATE));
        FILTER_END
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "Sharpen Filter Error: " << e << "\nAborting.\n";
        exit(EXIT_FAILURE);
    }
}

void boxFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        FILTER_START
        NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), size, offset,
            deviceDst.data(), deviceDst.pitch(), size,
            {3, 3}, {1, 1}, NPP_BORDER_REPLICATE));
        FILTER_END
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "Box Filter Error: " << e << "\nAborting.\n";
        exit(EXIT_FAILURE);
    }
}

void laplacianFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        FILTER_START
        NPP_CHECK_NPP(nppiFilterLaplaceBorder_8u_C1R(
            deviceSrc.data(), deviceSrc.pitch(), size, offset,
            deviceDst.data(), deviceDst.pitch(), size,
            NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE));
        FILTER_END
    }
    catch (const npp::Exception &e)
    {
        std::cerr << "Laplacian Filter Error: " << e << "\nAborting.\n";
        exit(EXIT_FAILURE);
    }
}

// Main filter selector
void applyFilter(const std::string &filterType, const std::string &filePath, const std::string &outputFile)
{
    if (filterType == "canny")
        cannyFilter(filePath, outputFile);
    else if (filterType == "sobel")
        sobelFilter(filePath, outputFile);
    else if (filterType == "gauss")
        gaussFilter(filePath, outputFile);
    else if (filterType == "sharpen")
        sharpenFilter(filePath, outputFile);
    else if (filterType == "box")
        boxFilter(filePath, outputFile);
    else if (filterType == "laplacian")
        laplacianFilter(filePath, outputFile);
    else
        std::cerr << "Unsupported filter: " << filterType << std::endl;

    cudaDeviceSynchronize();
    cudaDeviceReset();
}

// Main Entry
int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    if (!printNPPInfo(argc, argv))
        return EXIT_SUCCESS;

    findCudaDevice(argc, (const char **)argv);

    if (argc < 4)
    {
        std::cerr << "Usage: <executable> <input_image> <filter_type> <output_path>\n";
        return EXIT_FAILURE;
    }

    std::string inputPath{argv[1]};
    std::string filterType{argv[2]};
    std::string outputPath{argv[3]};

    if (!std::filesystem::exists(inputPath))
    {
        std::cerr << "Input file/folder does not exist.\n";
        return EXIT_FAILURE;
    }

    // If input is a file
    if (!std::filesystem::is_directory(inputPath))
    {
        std::string outputFile = outputPath;
        if (std::filesystem::is_directory(outputPath))
        {
            auto filename = std::filesystem::path(inputPath).filename().string();
            auto nameParts = splitString(filename, '.');
            outputFile = outputPath + "/" + nameParts.front() + ".bmp";
        }
        applyFilter(filterType, inputPath, outputFile);
    }

    return EXIT_SUCCESS;
}
