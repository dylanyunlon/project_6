/**
 * @brief Libinfer the fast cuda library for inference.
 * @file ixinfer.h
 */

#pragma GCC visibility push(default)
#if !defined(CUINFER_H_)
#define CUINFER_H_

/// Libinfer version major = 7.
#define CUINFER_MAJOR 7
/// Libinfer version minor = 6.
#define CUINFER_MINOR 6
/// Libinfer version patchlevel = 5.
#define CUINFER_PATCHLEVEL 5

/// Libinfer version = ::CUINFER_MAJOR * 1000 + ::CUINFER_MINOR * 100 +
/// ::CUINFER_PATCHLEVEL
#define CUINFER_VERSION                                                        \
  (CUINFER_MAJOR * 1000 + CUINFER_MINOR * 100 + CUINFER_PATCHLEVEL)

/// Libinfer priv version major = 3.
#define CUINFER_PRIV_MAJOR 3
/// Libinfer priv version minor = 3.
#define CUINFER_PRIV_MINOR 3
/// Libinfer priv version patch = 0.
#define CUINFER_PRIV_PATCH 0

/// Libinfer priv version = ::CUINFER_PRIV_MAJOR * 1000 + ::CUINFER_PRIV_MINOR *
/// 100 + ::CUINFER_PRIV_PATCH
#define CUINFER_PRIV_VERSION                                                   \
  (CUINFER_PRIV_MAJOR * 1000 + CUINFER_PRIV_MINOR * 100 + CUINFER_PRIV_PATCH)

#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdint.h>

#ifndef CUINFERWINAPI
#ifdef _WIN32
#define CUINFERWINAPI __stdcall
#else
#define CUINFERWINAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

struct cuinferContext;
/// @brief ::cuinferHandle_t is a point of struct to store ixinfer internal
/// info, e.g stream info.
/// @details The ::cuinferHandle_t is used in many cuinfer APIs. It must be
/// created with ::cuinferCreate before use and be destroyed after use by
/// ::cuinferDestroy.
/// @see ::cuinferCreate, ::cuinferDestroy
typedef struct cuinferContext *cuinferHandle_t;

/// @brief Return current cuinfer version.
/// @return ::CUINFER_VERSION
size_t CUINFERWINAPI cuinferGetVersion(void);

/// Returns CUDA Runtime version statically linked against cuinfer.
size_t CUINFERWINAPI cuinferGetCudartVersion(void);

/// Infer return status.
typedef enum {
  CUINFER_STATUS_SUCCESS = 0,         ///< Success. Everything goes well.
  CUINFER_STATUS_NOT_INITIALIZED = 1, ///< Nullptr or struct not initilized.
  CUINFER_STATUS_ALLOC_FAILED = 2,    ///< Memory allocation falied.
  CUINFER_STATUS_BAD_PARAM =
      3, ///< Bad parameters or bad combination of parameters.
  CUINFER_STATUS_INTERNAL_ERROR =
      4, ///< Internal error, which should not happen. Should be fixed.
  CUINFER_STATUS_INVALID_VALUE = 5, ///< Invalid single value.
  CUINFER_STATUS_ARCH_MISMATCH =
      6, ///< Libinfer is built for specific target, i.e. MR. Runing MR code on
         ///< BI will raise this error.
  CUINFER_STATUS_MAPPING_ERROR = 7,    ///< Not used.
  CUINFER_STATUS_EXECUTION_FAILED = 8, ///< Cuda api execution failed.
  CUINFER_STATUS_NOT_SUPPORTED = 9,    ///< Under development or not supported.
  CUINFER_STATUS_LICENSE_ERROR = 10,   ///< License error.
  CUINFER_STATUS_RUNTIME_PREREQUISITE_MISSING = 11, ///< Not used.
  CUINFER_STATUS_RUNTIME_IN_PROGRESS = 12,          ///< Not used.
  CUINFER_STATUS_RUNTIME_FP_OVERFLOW = 13,          ///< Not used.
} cuinferStatus_t;

/// @brief Return human-readable error messages.
/// @param[in] status The status to inspect.
/// @return Explaination to the status.
const char *CUINFERWINAPI cuinferGetErrorString(cuinferStatus_t status);

#ifndef __LIBRARY_TYPES_H__

/// Library property types.
typedef enum libraryPropertyType_t {
  MAJOR_VERSION,
  MINOR_VERSION,
  PATCH_LEVEL,
} libraryPropertyType;

#endif

/// @brief Get libraryPropertyType.
/// @param[in] type Library property type to query.
/// @param[out] value Correspond return value.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If out of range.
cuinferStatus_t CUINFERWINAPI cuinferGetProperty(libraryPropertyType type,
                                                 int *value);

/// @brief Create a libinfer handle.
/// @note This handle use the default \p cudaStream_t 0, which is synchroized
/// before and after other all other cuda operations. Use ::cuinferSetStream to
/// custom cuinfer stream to interleave compute and memory operations.
/// @note ::cuinferDestroy should be used to destoy a \p handle.
/// @param[out] handle The pointer to handle.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p handle is null.
/// * ::CUINFER_STATUS_ALLOC_FAILED If alloc failed.
cuinferStatus_t CUINFERWINAPI cuinferCreate(cuinferHandle_t *handle);

/// @brief Destroy a libinfer handle.
/// @param[in] handle The handle to destory.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_INTERNAL_ERROR If internal error happened.
cuinferStatus_t CUINFERWINAPI cuinferDestroy(cuinferHandle_t handle);

/// @brief Set a \p cudaStream_t to a \p handle.
/// @details All operation associated with this \p handle will use this p
/// @param[in] handle The target ::cuinferHandle_t.
/// @param[in] streamId The new \p cudaStream_t to put.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If handle is null.
/// * ::CUINFER_STATUS_INTERNAL_ERROR If internal error happened.
cuinferStatus_t CUINFERWINAPI cuinferSetStream(cuinferHandle_t handle,
                                               cudaStream_t streamId);

/// @brief Get a \p cudaStream_t corresponding to a \p handle.
/// @param[in] handle The target ::cuinferHandle_t.
/// @param[out] streamId The \p cudaStream_t to get.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If handle is null.
/// * ::CUINFER_STATUS_INTERNAL_ERROR If internal error happened.
cuinferStatus_t CUINFERWINAPI cuinferGetStream(cuinferHandle_t handle,
                                               cudaStream_t *streamId);

/// @brief Pointer to tensor descriptions.
/// @details Contains tensor ::cuinferTensorFormat_t, strides and dimensions
/// infos.
/// @see ::cuinferCreateTensorDescriptor, ::cuinferDestroyTensorDescriptor,
/// ::cuinferSetTensor4dDescriptor, ::cuinferSetTensor4dDescriptorEx,
/// ::cuinferSetTensorNdDescriptor, ::cuinferSetTensorNdDescriptorEx,
/// ::cuinferGetTensor4dDescriptor, ::cuinferGetTensorNdDescriptor and
/// ::cuinferGetTensorSizeInBytes.
typedef struct cuinferTensorStruct *cuinferTensorDescriptor_t;

/// @brief Pointer to convolution descriptions.
/// @details Contains padding, stride, dilation, ::cuinferConvolutionMode_t,
/// ::cuinferDataType_t, ::cuinferMathType_t and group_count infos.
/// @see ::cuinferCreateConvolutionDescriptor,
/// ::cuinferDestroyConvolutionDescriptor, ::cuinferSetConvolutionGroupCount,
/// ::cuinferSetConvolution2dDescriptor, ::cuinferSetConvolutionNdDescriptor,
/// ::cuinferGetConvolutionMathType, ::cuinferGetConvolutionGroupCount,
/// ::cuinferGetConvolution2dDescriptor,
/// ::cuinferGetConvolution2dForwardOutputDim,
/// ::cuinferGetConvolutionNdDescriptor and
/// ::cuinferGetConvolutionNdForwardOutputDim.
typedef struct cuinferConvolutionStruct *cuinferConvolutionDescriptor_t;

/// @brief Pointer to pooling layer descriptions.
/// @details Contains ::cuinferPoolingMode_t, ::cuinferNanPropagation_t,
/// window_dim, padding and stride infos.
/// @see ::cuinferCreatePoolingDescriptor, ::cuinferDestroyPoolingDescriptor,
/// ::cuinferSetPooling2dDescriptor, ::cuinferSetPoolingNdDescriptor,
/// ::cuinferGetPooling2dDescriptor, ::cuinferGetPoolingNdDescriptor,
/// ::cuinferGetPoolingNdForwardOutputDim and
/// ::cuinferGetPooling2dForwardOutputDim.
typedef struct cuinferPoolingStruct *cuinferPoolingDescriptor_t;

/// @brief Pointer to filter tensor descriptions.
/// @details Contains ::cuinferDataType_t, ::cuinferTensorFormat_t and
/// dimentions infos.
/// @see ::cuinferCreateFilterDescriptor, ::cuinferDestroyFilterDescriptor,
/// ::cuinferSetFilter4dDescriptor, ::cuinferSetFilterNdDescriptor,
/// ::cuinferGetFilter4dDescriptor and ::cuinferGetFilterNdDescriptor.
typedef struct cuinferFilterStruct *cuinferFilterDescriptor_t;

/// @brief Pointer to LRN(Learning Resource Network) descriptions.
/// @details Contains LRN's \p n, \p alpha, \p beta ane \p k infos.
/// @see ::cuinferCreateLRNDescriptor, ::cuinferDestroyLRNDescriptor,
/// ::cuinferSetLRNDescriptor and ::cuinferGetLRNDescriptor.
typedef struct cuinferLRNStruct *cuinferLRNDescriptor_t;

/// @brief Pointer to activation descriptions.
/// @details Contains ::cuinferActivationMode_t, ::cuinferNanPropagation_t and
/// coef infos.
/// @note The coef can mean different param in different
/// ::cuinferActivationMode_t, i.e. ceiling for clipped RELU, alpha for ELU.
/// @see ::cuinferCreateActivationDescriptor,
/// ::cuinferDestroyActivationDescriptor, ::cuinferSetActivationDescriptor and
/// ::cuinferGetActivationDescriptor.
typedef struct cuinferActivationStruct *cuinferActivationDescriptor_t;

/// @brief Pointer to reduce tensor descriptions.
/// @details Contains ::cuinferReduceTensorOp_t, ::cuinferDataType_t,
/// ::cuinferNanPropagation_t, ::cuinferReduceTensorIndices_t and
/// ::cuinferIndicesType_t infos.
/// @see ::cuinferCreateReduceTensorDescriptor,
/// ::cuinferCreateReduceTensorDescriptor and
/// ::cuinferSetReduceTensorDescriptor.
typedef struct cuinferReduceTensorStruct *cuinferReduceTensorDescriptor_t;

/// @brief Pointer to CTC(Connectionist temporal classification) loss
/// descriptions.
/// @details Contains ::cuinferDataType_t, ::cuinferLossNormalizationMode_t and
/// ::cuinferNanPropagation_t.
/// @see ::cuinferCreateCTCLossDescriptor, ::cuinferDestroyCTCLossDescriptor,
/// ::cuinferSetCTCLossDescriptor, ::cuinferSetCTCLossDescriptorEx,
/// ::cuinferGetCTCLossDescriptor and ::cuinferGetCTCLossDescriptorEx.
typedef struct cuinferCTCLossStruct *cuinferCTCLossDescriptor_t;

/// Libinfer data types.
typedef enum {
  CUINFER_DATA_FLOAT = 0,  ///< 32-bit ieee float type.
  CUINFER_DATA_DOUBLE = 1, ///< 64-bit ieee double float type.
  CUINFER_DATA_HALF = 2,   ///< 16-bit ieee half float type.
  CUINFER_DATA_INT8 = 3,   ///< 8-bit signed integer type.
  CUINFER_DATA_INT32 = 4,  ///< 32-bit signed integer type.
  CUINFER_DATA_INT8x4 = 5, ///< 4x8-bit signed integer type. Aligned to 4 bytes.
  CUINFER_DATA_UINT8 = 6,  ///< 8-bit unsigned integer type.
  CUINFER_DATA_UINT8x4 =
      7, ///< 4x8-bit unsigned integer type. Aligned to 4 bytes.
  CUINFER_DATA_INT8x32 =
      8, ///< 32x8-bit signed integer type. Aligned to 32 bytes.
  CUINFER_DATA_BFLOAT16 = 9, ///< Google's brain floating point. 16-bit.
} cuinferDataType_t;

/// Libinfer math type.
typedef enum {
  CUINFER_DEFAULT_MATH = 0,                    ///< Default math type.
  CUINFER_TENSOR_OP_MATH = 1,                  ///< Perffer to use tensor op.
  CUINFER_TENSOR_OP_MATH_ALLOW_CONVERSION = 2, ///< Not used.
} cuinferMathType_t;

/// @brief Libinfer propagate NaN(not a number) option. @details
/// ::cuinferNanPropagation_t is used to indicate if a float number result in
/// NaN(Not a Number) should be propagate nan or not (0 will be propagated
/// instead).This setting is only useful for float type computation. This is
/// used in setting ::cuinferReduceTensorDescriptor_t,
/// ::cuinferPoolingDescriptor_t, ::cuinferActivationDescriptor_t,
/// ::cuinferRNNDescriptor_t and ::cuinferCTCLossDescriptor_t.
typedef enum {
  CUINFER_NOT_PROPAGATE_NAN = 0, ///< \p 0 will be propagating for \p NaN and \p
                                 ///< Inf values in float types.
  CUINFER_PROPAGATE_NAN =
      1, ///< \p NaN and \p Inf will be propagating in float types.
} cuinferNanPropagation_t;

/// Is algorithm result determinstic(same input always produce same outputs).
typedef enum {
  CUINFER_NON_DETERMINISTIC = 0, ///< Same input may poduce different outputs.
                                 ///< Due to data race, i.e. atomic operations.
  CUINFER_DETERMINISTIC = 1,     ///< Same input always produce same outputs.
} cuinferDeterminism_t;

/// Maximum supported number of tensor dimensions.
#define CUINFER_DIM_MAX 8

/// @brief Create an instance of a generic Tensor descriptor.
/// @note ::cuinferDestroyTensorDescriptor should be called after use.
/// @param[out] tensorDesc Pointer to store ::cuinferTensorDescriptor_t.
cuinferStatus_t CUINFERWINAPI
cuinferCreateTensorDescriptor(cuinferTensorDescriptor_t *tensorDesc);

/// @brief Tensor format stored in memory.
/// @details
/// * ::CUINFER_TENSOR_NCHW tensor runs faster in CPUs.
/// * ::CUINFER_TENSOR_NHWC tensor runs faster in GPUs.
/// * ::CUINFER_TENSOR_NCHW_VECT_C split dim C and run faster in both.
typedef enum {
  CUINFER_TENSOR_NCHW = 0,
  ///< Elements are stored in batch, channel, depth(3d only), height and
  ///< weight order(higher to lower).
  CUINFER_TENSOR_NHWC =
      1, ///< Elements are stored in batch, depth(3d only),
         ///< height, weight and channel order(higher to lower).
  CUINFER_TENSOR_NCHW_VECT_C = 2,
  ///< Elements are stored in batch, channel / 4, depth(3d only), height,
  ///< weight, 4 order(higher to lower), where channel is split by 4 into 2
  ///< parts.
} cuinferTensorFormat_t;

/// @brief Setup params for ::cuinferTensorDescriptor_t.
/// @param[out] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @param[in] format Tensor format.
/// @param[in] dataType Tensor data type.
/// @param[in] n Tensor batch size.
/// @param[in] c Tensor channel size.
/// @param[in] h Tensor height.
/// @param[in] w Tensor width.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param out of range or \p tensorDesc is null
/// or c is not multiple of 4 in ::CUINFER_TENSOR_NCHW_VECT_C.
cuinferStatus_t CUINFERWINAPI cuinferSetTensor4dDescriptor(
    cuinferTensorDescriptor_t tensorDesc, cuinferTensorFormat_t format,
    cuinferDataType_t dataType, int n, int c, int h, int w);

/// @brief Setup params for ::cuinferTensorDescriptor_t.
/// @param[out] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @param[in] dataType Tensor data type.
/// @param[in] n Tensor batch size.
/// @param[in] c Tensor channel size.
/// @param[in] h Tensor height.
/// @param[in] w Tensor width.
/// @param[in] nStride Stride of batch.
/// @param[in] cStride Stride of channel.
/// @param[in] hStride Stride of height.
/// @param[in] wStride Stride of width.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param out of range or \p tensorDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferSetTensor4dDescriptorEx(
    cuinferTensorDescriptor_t tensorDesc, cuinferDataType_t dataType, int n,
    int c, int h, int w, int nStride, int cStride, int hStride, int wStride);

/// @brief Return params for ::cuinferTensorDescriptor_t.
/// @param[in] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @param[out] dataType Tensor data type.
/// @param[out] n Tensor batch size.
/// @param[out] c Tensor channel size.
/// @param[out] h Tensor height.
/// @param[out] w Tensor width.
/// @param[out] nStride Stride of batch.
/// @param[out] cStride Stride of channel.
/// @param[out] hStride Stride of height.
/// @param[out] wStride Stride of width.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p tensorDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetTensor4dDescriptor(
    const cuinferTensorDescriptor_t tensorDesc, cuinferDataType_t *dataType,
    int *n, int *c, int *h, int *w, int *nStride, int *cStride, int *hStride,
    int *wStride);

/// @brief Setup params for 2d/3d ::cuinferTensorDescriptor_t.
/// @details The input order(dim0/stride0, dim1/stride1, ...) is batch, channel,
/// depth(3d only), height and weight.
/// @note The ::CUINFER_TENSOR_NHWC format may change the strides.
/// @note Can not set ::CUINFER_TENSOR_NCHW_VECT_C format.
/// @see ::cuinferTensorFormat_t, ::cuinferSetTensorNdDescriptorEx
/// @param[out] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @param[in] dataType Tensor data type.
/// @param[in] nbDims Number of dimensions. 4 for 2d conv and 5 for 3d conv.
/// @param[in] dimA Size of each dimension.Nchw for 2d and ncdhw for 3d.
/// @param[in] strideA Stride of each dimension. Nchw for 2d and ncdhw for 3d.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If \p nbDims not in range[4,
/// ::CUINFER_DIM_MAX].
/// * ::CUINFER_STATUS_BAD_PARAM If \p tensorDesc is null or invalid dims.
cuinferStatus_t CUINFERWINAPI cuinferSetTensorNdDescriptor(
    cuinferTensorDescriptor_t tensorDesc, cuinferDataType_t dataType,
    int nbDims, const int dimA[], const int strideA[]);

/// @brief Setup params for 2d/3d ::cuinferTensorDescriptor_t.
/// @details The input order(dim0/stride0, dim1/stride1, ...) is batch, channel,
/// depth(3d only), height and weight.
/// @note Strides is set according to \p format and \p nbDims.
/// @see ::cuinferTensorFormat_t, ::cuinferSetTensorNdDescriptor
/// @param[out] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @param[in] format Tensor format.
/// @param[in] dataType Tensor data type.
/// @param[in] nbDims Number of dimensions. 4 for 2d conv and 5 for 3d conv.
/// @param[in] dimA Size of each dimension.Nchw for 2d and ncdhw for 3d.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If \p nbDims not in range[4,
/// ::CUINFER_DIM_MAX].
/// * ::CUINFER_STATUS_BAD_PARAM If \p tensorDesc is null or invalid dims.
cuinferStatus_t CUINFERWINAPI cuinferSetTensorNdDescriptorEx(
    cuinferTensorDescriptor_t tensorDesc, cuinferTensorFormat_t format,
    cuinferDataType_t dataType, int nbDims, const int dimA[]);

/// @brief Return params for 2d/3d ::cuinferTensorDescriptor_t.
/// @details The output order(dim0/stride0, dim1/stride1, ...) is batch,
/// channel, depth(3d only), height and weight.
/// @see cuinferSetTensorNdDescriptor
/// @param[in] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @param[out] nbDimsRequested Not used. @todo \p nbDimsRequested not used.
/// @param[out] dataType Tensor data type.
/// @param[out] nbDims Number of dimensions. 4 for 2d conv and 5 for 3d conv.
/// @param[out] dimA Size of each dimension.Nchw for 2d and ncdhw for 3d.
/// @param[out] strideA Stride of each dimension. Nchw for 2d and ncdhw for 3d.
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p tensorDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetTensorNdDescriptor(
    const cuinferTensorDescriptor_t tensorDesc, int nbDimsRequested,
    cuinferDataType_t *dataType, int *nbDims, int dimA[], int strideA[]);

/// @brief Returns psysical space needed by a tensor.
/// @note The psysical space needed can be slightly larger than logical space
/// due to stride sittings(padding).
/// @param[in] tensorDesc  Pointer to target ::cuinferTensorDescriptor_t.
/// @param[out] size Result size in bytes.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p tensorDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetTensorSizeInBytes(
    const cuinferTensorDescriptor_t tensorDesc, size_t *size);

/// Destroy an instance of Tensor4d descriptor

/// @brief Destroy an instance of ::cuinferTensorDescriptor_t.
/// @param[in] tensorDesc Pointer to target ::cuinferTensorDescriptor_t.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_INTERNAL_ERROR
cuinferStatus_t CUINFERWINAPI
cuinferDestroyTensorDescriptor(cuinferTensorDescriptor_t tensorDesc);

/// @brief Tensor layout conversion helper y = alpha * x + beta * y.
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor in host memory. Type is always
/// float for now.
/// @param[in] xDesc Meta info of tensor x.
/// @param[in] x Input tensor data.
/// @param[in] beta Pointer to scaling factor in host memory. Type is always
/// float for now.
/// @param[in] yDesc Meta info of tensor y.
/// @param[in,out] y Input and output tensor data.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If bad params.
/// * ::CUINFER_STATUS_INTERNAL_ERROR If internal error happened.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If algo not supported.
cuinferStatus_t CUINFERWINAPI cuinferTransformTensor(
    cuinferHandle_t handle, const void *alpha,
    const cuinferTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cuinferTensorDescriptor_t yDesc, void *y);

/// @brief Add two Tensor. C = alpha * A + beta * C.
/// @todo difference to ::cuinferTransformTensor?
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor in host memory. Type is always
/// float for now.
/// @param[in] aDesc The tensor descripter of A.
/// @param[in] A Const pointer to tensor data A.
/// @param[in] beta Pointer to scaling factor in host memory. Type is always
/// float for now.
/// @param[in] cDesc The tensor descripter of C.
/// @param[in,out] C Input and output tensor data C.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If bad params.
/// * ::CUINFER_STATUS_INTERNAL_ERROR If internal error happened.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If algo not supported.
cuinferStatus_t CUINFERWINAPI cuinferAddTensor(
    cuinferHandle_t handle, const void *alpha,
    const cuinferTensorDescriptor_t aDesc, const void *A, const void *beta,
    const cuinferTensorDescriptor_t cDesc, void *C);

/// Libinfer ReduceTensor op type.
typedef enum {
  CUINFER_REDUCE_TENSOR_ADD = 0,   ///< Addition.
  CUINFER_REDUCE_TENSOR_MUL = 1,   ///< Multiplication.
  CUINFER_REDUCE_TENSOR_MIN = 2,   ///< Minimum.
  CUINFER_REDUCE_TENSOR_MAX = 3,   ///< Maximum.
  CUINFER_REDUCE_TENSOR_AMAX = 4,  ///< Argmax. The index of Maximum element.
  CUINFER_REDUCE_TENSOR_AVG = 5,   ///< Average. \f$ \frac{\sum{x}}{n} \f$
  CUINFER_REDUCE_TENSOR_NORM1 = 6, ///< Absolute-value norm. \f$ \sum{|x|} \f$
  CUINFER_REDUCE_TENSOR_NORM2 = 7, ///< Euclidean norm. \f$ \sqrt{\sum{x^2}} \f$
  CUINFER_REDUCE_TENSOR_MUL_NO_ZEROS =
      8, ///< Multiplication only to valid values.
} cuinferReduceTensorOp_t;

/// Not used.
typedef enum {
  CUINFER_REDUCE_TENSOR_NO_INDICES = 0,
  CUINFER_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} cuinferReduceTensorIndices_t;

/// Not used.
typedef enum {
  CUINFER_32BIT_INDICES = 0,
  CUINFER_64BIT_INDICES = 1,
  CUINFER_16BIT_INDICES = 2,
  CUINFER_8BIT_INDICES = 3,
} cuinferIndicesType_t;

/// @brief Create a ::cuinferReduceTensorDescriptor_t.
/// @param[out] reduceTensorDesc Pointer to ::cuinferReduceTensorDescriptor_t.
/// @return
/// * ::CUINFER_STATUS_SUCCESS if success.
/// * ::CUINFER_STATUS_ALLOC_FAILED if malloc failed.
cuinferStatus_t CUINFERWINAPI cuinferCreateReduceTensorDescriptor(
    cuinferReduceTensorDescriptor_t *reduceTensorDesc);

/// @brief Set a ::cuinferReduceTensorDescriptor_t.
/// Not used.
/// @param[out] reduceTensorDesc The target ::cuinferReduceTensorDescriptor_t.
/// @param[in] reduceTensorOp The resuce tensor Op.
/// @param[in] reduceTensorCompType The reduce tensor compute type.
/// @param[in] reduceTensorNanOpt The reduce tensor op NaN propgation setting.
/// @param[in] reduceTensorIndices Not used.
/// @param[in] reduceTensorIndicesType Not used.
/// @return
/// * ::CUINFER_STATUS_SUCCESS if success.
/// * ::CUINFER_STATUS_BAD_PARAM if \p reduceTensorDesc is null or bad param.
cuinferStatus_t CUINFERWINAPI cuinferSetReduceTensorDescriptor(
    cuinferReduceTensorDescriptor_t reduceTensorDesc,
    cuinferReduceTensorOp_t reduceTensorOp,
    cuinferDataType_t reduceTensorCompType,
    cuinferNanPropagation_t reduceTensorNanOpt,
    cuinferReduceTensorIndices_t reduceTensorIndices,
    cuinferIndicesType_t reduceTensorIndicesType);

/// @todo Not used?
cuinferStatus_t CUINFERWINAPI cuinferReduceTensor(
    cuinferHandle_t handle,
    const cuinferReduceTensorDescriptor_t reduceTensorDesc, void *indices,
    size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const cuinferTensorDescriptor_t aDesc, const void *A,
    const void *beta, const cuinferTensorDescriptor_t cDesc, void *C);

/// @brief Convolution mode. @details They do the same computation while data
/// layout is different.
typedef enum {
  /// Convolution. Take 2d for example \f$
  /// y[i,j]=\sum{x[i,j]w[\mathrm{height}-1-i,\mathrm{weight}-1-j]} \f$.
  CUINFER_CONVOLUTION = 0,
  /// Cross correlation. Take 2d for example \f$ y[i,j]=\sum{x[i,j]w[i,j]} \f$.
  CUINFER_CROSS_CORRELATION = 1,
} cuinferConvolutionMode_t;

/// @brief Create a ::cuinferFilterDescriptor_t.
/// @param[out] filterDesc The descriptor for the filter created.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_ALLOC_FAILED If allocation failed.
cuinferStatus_t CUINFERWINAPI
cuinferCreateFilterDescriptor(cuinferFilterDescriptor_t *filterDesc);

/// @brief Set a 4d ::cuinferFilterDescriptor_t.
/// @param[out] filterDesc The pointer to target ::cuinferFilterDescriptor_t.
/// @param[in] dataType The data type of the filter.
/// @param[in] format The format of the filter.
/// @param[in] k Number of filters.
/// @param[in] c Number of input channels.
/// @param[in] h Filter height.
/// @param[in] w Filter weight.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If invalid param.
cuinferStatus_t CUINFERWINAPI cuinferSetFilter4dDescriptor(
    cuinferFilterDescriptor_t filterDesc, cuinferDataType_t dataType,
    cuinferTensorFormat_t format, int k, int c, int h, int w);

/// @brief Get info form 4d ::cuinferFilterDescriptor_t.
/// @param[in] filterDesc The pointer to target ::cuinferFilterDescriptor_t.
/// @param[out] dataType The data type of the filter.
/// @param[out] format The format of the filter.
/// @param[out] k Number of filters.
/// @param[out] c Number of input channels.
/// @param[out] h Filter height.
/// @param[out] w Filter weight.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p filterDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetFilter4dDescriptor(
    const cuinferFilterDescriptor_t filterDesc, cuinferDataType_t *dataType,
    cuinferTensorFormat_t *format, int *k, int *c, int *h, int *w);

/// @brief Set a ::cuinferFilterDescriptor_t.
/// @see ::cuinferGetFilter4dDescriptor
/// @param[out] filterDesc The pointer to target ::cuinferFilterDescriptor_t.
/// @param[in] dataType The datatype of the filter.
/// @param[in] format The format of the filter.
/// @param[in] nbDims Number of dimensions, 4 or 5.
/// @param[in] filterDimA Starting from index 0; k, c, h, w for 4d and k, c, d,
/// h, w for 5d.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p filterDesc is null or bad params.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If type is not supported.
cuinferStatus_t CUINFERWINAPI cuinferSetFilterNdDescriptor(
    cuinferFilterDescriptor_t filterDesc, cuinferDataType_t dataType,
    cuinferTensorFormat_t format, int nbDims, const int filterDimA[]);

/// @brief Get info from a ::cuinferFilterDescriptor_t.
/// @see ::cuinferGetFilter4dDescriptor
/// @param[in] filterDesc The pointer to target ::cuinferFilterDescriptor_t.
/// @param[out] nbDimsRequested Not used.
/// @param[out] dataType The datatype of the filter.
/// @param[out] format The format of the filter.
/// @param[out] nbDims Number of dimensions, 4 or 5.
/// @param[out] filterDimA Starting from index 0; k, c, h, w for 4d and k, c, d,
/// h, w for 5d.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p filterDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetFilterNdDescriptor(
    const cuinferFilterDescriptor_t filterDesc, int nbDimsRequested,
    cuinferDataType_t *dataType, ///< image data type
    cuinferTensorFormat_t *format, int *nbDims, int filterDimA[]);

/// @brief Return bytes used by a ::cuinferFilterDescriptor_t.
/// @param[in] filterDesc The pointer to target ::cuinferFilterDescriptor_t.
/// @param[out] size The pysical size in bytes.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p filterDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetFilterSizeInBytes(
    const cuinferFilterDescriptor_t filterDesc, size_t *size);

/// @brief Destopy a ::cuinferFilterDescriptor_t after use.
/// @see ::cuinferCreateFilterDescriptor
/// @param[in] filterDesc The pointer to target ::cuinferFilterDescriptor_t.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI
cuinferDestroyFilterDescriptor(cuinferFilterDescriptor_t filterDesc);

/// @brief Create an instance of ::cuinferConvolutionDescriptor_t.
/// @param[out] convDesc The pointer to store ::cuinferConvolutionDescriptor_t.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_ALLOC_FAILED If allocation failed.
cuinferStatus_t CUINFERWINAPI
cuinferCreateConvolutionDescriptor(cuinferConvolutionDescriptor_t *convDesc);

/// @brief Set the \p mathType for a ::cuinferConvolutionDescriptor_t.
/// @param[out] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[in] mathType The target ::cuinferMathType_t.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p convDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferSetConvolutionMathType(
    cuinferConvolutionDescriptor_t convDesc, cuinferMathType_t mathType);

/// @brief Get the \p mathType for a ::cuinferConvolutionDescriptor_t.
/// @param[in] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[out] mathType The target ::cuinferMathType_t.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p convDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionMathType(
    cuinferConvolutionDescriptor_t convDesc, cuinferMathType_t *mathType);

/// @brief Set the \p groupCount for a ::cuinferConvolutionDescriptor_t.
/// @param[out] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[in] groupCount The target group count.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p convDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferSetConvolutionGroupCount(
    cuinferConvolutionDescriptor_t convDesc, int groupCount);

/// @brief Get the \p groupCount for a ::cuinferConvolutionDescriptor_t.
/// @param[in] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[out] groupCount The target group count.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p convDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionGroupCount(
    cuinferConvolutionDescriptor_t convDesc, int *groupCount);

/// @brief Set a 2d ::cuinferConvolutionDescriptor_t.
/// @param[out] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[in] pad_h The padding of data in height.
/// @param[in] pad_w The padding of data in weight.
/// @param[in] u The stride in filter in height.
/// @param[in] v The stride in filter in weight.
/// @param[in] dilation_h The filter dilation in height.
/// @param[in] dilation_w The filter dilation in weight.
/// @param[in] mode The convolution mode.
/// @param[in] computeType The datatype in compute. Can be different to input
/// and output datatype.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param is not valid.
cuinferStatus_t CUINFERWINAPI cuinferSetConvolution2dDescriptor(
    cuinferConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v,
    int dilation_h, int dilation_w, cuinferConvolutionMode_t mode,
    cuinferDataType_t computeType);

/// @brief Return the info from a 2d ::cuinferConvolutionDescriptor_t.
/// @param[in] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[out] pad_h The padding of data in height.
/// @param[out] pad_w The padding of data in weight.
/// @param[out] u The stride in filter in height.
/// @param[out] v The stride in filter in weight.
/// @param[out] dilation_h The filter dilation in height.
/// @param[out] dilation_w The filter dilation in weight.
/// @param[out] mode The convolution mode.
/// @param[out] computeType The datatype in compute. Can be different to input
/// and output datatype.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param is \p convDesc is null.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolution2dDescriptor(
    const cuinferConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w,
    int *u, int *v, int *dilation_h, int *dilation_w,
    cuinferConvolutionMode_t *mode, cuinferDataType_t *computeType);

/// Helper function to return the dimensions of the output tensor given a
/// convolution descriptor

/// @brief Helper function to calculate the result dimensions given a
/// ::cuinferConvolutionDescriptor_t and input ::cuinferTensorDescriptor_t.
/// @param[in] convDesc The conv descriptor.
/// @param[in] inputTensorDesc The input tensor descriptor.
/// @param[in] filterDesc The filter descriptor.
/// @param[out] n The batch number of result tensor.
/// @param[out] c The number of channels of result tensor.
/// @param[out] h The height of result tensor.
/// @param[out] w The weight of result tensor.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param is \p convDesc is null or invalid
/// conbination of params in \p convDesc, \p inputTensorDesc and \p filterDesc.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolution2dForwardOutputDim(
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t inputTensorDesc,
    const cuinferFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w);

/// @brief Set a 2d or 3d ::cuinferConvolutionDescriptor_t.
/// @param[out] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[in] arrayLength The input array length, 2 for 2d, 3 for 3d.
/// @param[in] padA The input padding array. Height, weight for 2d; depth,
/// height, weight for 3d.
/// @param[in] filterStrideA The filter stride array. Height, weight for 2d;
/// depth, height, weight for 3d.
/// @param[in] dilationA The filter dilation array. Height, weight for 2d;
/// depth, height, weight for 3d.
/// @param[in] mode The convolution mode.
/// @param[in] computeType The datatype in compute. Can be different to input
/// and output datatype.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param is \p convDesc is null or invalid
/// conbination of params in \p convDesc, \p inputTensorDesc and \p filterDesc.
cuinferStatus_t CUINFERWINAPI cuinferSetConvolutionNdDescriptor(
    cuinferConvolutionDescriptor_t convDesc, int arrayLength, const int padA[],
    const int filterStrideA[], const int dilationA[],
    cuinferConvolutionMode_t mode, cuinferDataType_t computeType);

/// @brief Set a 2d or 3d ::cuinferConvolutionDescriptor_t.
/// @param[in] convDesc The target ::cuinferConvolutionDescriptor_t.
/// @param[in] arrayLengthRequested Not used.
/// @param[out] arrayLength The array length. 2 for 2d and 3 for 3d.
/// @param[out] padA The input padding array. Height, weight for 2d; depth,
/// height, weight for 3d.
/// @param[out] strideA The filter stride array. Height, weight for 2d;
/// depth, height, weight for 3d.
/// @param[out] dilationA The filter dilation array. Height, weight for 2d;
/// depth, height, weight for 3d.
/// @param[out] mode The convolution mode.
/// @param[out] computeType The datatype in compute. Can be different to input
/// and output datatype.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If param is \p convDesc is null or invalid
/// conbination of params in \p convDesc, \p inputTensorDesc and \p filterDesc.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionNdDescriptor(
    const cuinferConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[], int strideA[], int dilationA[],
    cuinferConvolutionMode_t *mode, cuinferDataType_t *computeType);

/// @brief Get the output dimensions given convolution descriptions.
/// @param[in] convDesc The convolution descriptor.
/// @param[in] inputTensorDesc The input tensor descriptor.
/// @param[in] filterDesc The filter descriptor.
/// @param[in] nbDims Number of dimensions. 2 for 2d and 3 for 3d.
/// @param[out] tensorOuputDimA The result output tensor dimensions. Height,
/// weight for 2d and depth, height, weight for 3d.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p convDesc is null or invalid conbination
/// of params in \p convDesc, \p inputTensorDesc and \p filterDesc.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionNdForwardOutputDim(
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t inputTensorDesc,
    const cuinferFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[]);

/// @brief Destroy a convolution descriptor after use.
/// @param[in] convDesc The ::cuinferConvolutionDescriptor_t to be destroyed.
/// @warning Deleting a \p convDesc twice is an undefined behavior.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI
cuinferDestroyConvolutionDescriptor(cuinferConvolutionDescriptor_t convDesc);

/**
 * @brief Function to concatenate a few tensors to a output tensor.
 * @details This function is made to concatenate tensors, the number of input
 tensors can be two,three, or four channel in cuinferTensorDescriptor_t is
 padded channel, the real channel is realc (when axis != 3, the realc is
 useless).
 *
 * Example: Concat two tensors, x1Desc = {N, H, W, padc1}, x2Desc = {N, H, W,
 padc2}, yDesc = {N, H, W, realc1 + realc2 + y_pad}.
 *
 * @param[in] x1Desc The information of input1.
 * @param[in] x1 Input1 address.
 * @param[in] x2Desc The information of input2.
 * @param[in] x2 Input2 address.
 * @param[in] x3Desc The information of input2.
 * @param[in] x3 Input3 address.
 * @param[in] x4Desc The information of input2.
 * @param[in] x4 Input4 address.
 * @param[in] yDesc The information of output.
 * @param[out] y Output address.
 * @param[in] axis Decide whether realc is useful.
 * @param[in] bQuant Decide whether the result need multiply \p y_scale and \p
 scale1 \p scale2 \p scale3 \p scale4.
 * @return
 * * ::CUINFER_STATUS_BAD_PARAM If \p x1, \p x2 is \p nullptr.
 * * ::CUINFER_STATUS_NOT_SUPPORTED If not supported.
 * * ::CUINFER_STATUS_SUCCESS If success.
*/
cuinferStatus_t CUINFERWINAPI cuinferConcatenate(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t x1Desc,
    const void *x1, const void *scale1, const int realc1,
    const cuinferTensorDescriptor_t x2Desc, const void *x2, const void *scale2,
    const int realc2, const cuinferTensorDescriptor_t x3Desc, const void *x3,
    const void *scale3, const int realc3,
    const cuinferTensorDescriptor_t x4Desc, const void *x4, const void *scale4,
    const int realc4, const cuinferTensorDescriptor_t yDesc, void *y,
    const void *y_scale, const int axis, bool bQuant);

/// @brief Split input int8 tensor to 2 or 3 tensors.
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param[in] height The height of the image tensor.
/// @param[in] width The width of the image tensor.
/// @param[in] sizeLen The split size, 2 or 3.
/// @param[in] sizes The size start of each parts.
/// @param[in] axis The axis to split.
/// @param[out] y The discriptor of output tensor y.
/// @return
/// * ::CUINFER_STATUS_NOT_SUPPORTED If not supported.
/// * ::CUINFER_STATUS_SUCCESS If success.
/// @todo Currently not used by other library.
cuinferStatus_t CUINFERWINAPI cuinferSplitForward(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const void *x, const int batch, const int height, const int width,
    const int sizeLen, const int *sizes, const int axis, void *y);

/// Interpolation method used in image resize.
typedef enum {
  CUINFER_INTER_NEAREST = 0, ///< Pixel is determined by it's nearest neighbor.
  CUINFER_INTER_LINEAR = 1,  ///< Pixel is determined by linear interpolation.
  CUINFER_INTER_CUBIC = 2,   ///< Pixcel is determined by cubic interpolation.
  CUINFER_INTER_AREA = 3,    ///< Not used.
} cuinferInterpolationFlag_t;

/// @todo Explain this.
typedef enum {
  CUINFER_HALF_PIXEL = 0,
  CUINFER_ALIGN_CORNERS = 1,
  CUINFER_ASYMMETRIC = 2,
} cuinferCoordinateTransformationMode_t;

/// @brief Resize a image.
/// @note The input pointer and output space are not overlap.
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The tensor descriptor of the input.
/// @param[in] x The const pointer of input.
/// @param[in] interpolation Interpolation mode.
/// @param[in] transformMode
/// @param[in] yDesc The tensor descriptor of the output.
/// @param[out] y The pointer of output tensor y.
/// @return
/// * ::CUINFER_STATUS_NOT_SUPPORTED If algo is not supported.
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI
cuinferResize2D(cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
                const void *x, cuinferInterpolationFlag_t interpolation,
                cuinferCoordinateTransformationMode_t transformMode,
                const cuinferTensorDescriptor_t yDesc, void *y);

/// Convolution forward algo selection preference.
typedef enum {
  CUINFER_CONVOLUTION_FWD_NO_WORKSPACE = 0,   ///< No extra workspace.
  CUINFER_CONVOLUTION_FWD_PREFER_FASTEST = 1, ///< Prefer fastest.
  CUINFER_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT =
      2, ///< Specify workspace limit.
} cuinferConvolutionFwdPreference_t;

/// Convolution forward algo.
typedef enum {
  CUINFER_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,         ///< Implicit gemm.
  CUINFER_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1, ///< Implicit
  CUINFER_CONVOLUTION_FWD_ALGO_GEMM = 2,                  ///< Gemm.
  CUINFER_CONVOLUTION_FWD_ALGO_DIRECT = 3, ///< Direct compute use cuda call.
  CUINFER_CONVOLUTION_FWD_ALGO_FFT = 4,    ///< FFT.
  CUINFER_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,        ///< FFT tiling.
  CUINFER_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,          ///< Winograd.
  CUINFER_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7, ///< Winograd nonfused.
  CUINFER_CONVOLUTION_FWD_ALGO_COUNT = 8,             ///< Total algo count.
} cuinferConvolutionFwdAlgo_t;

/// How to connect conv result and previous result.
typedef enum {
  CUINFER_CONNECTION_NONE = 0,   ///< No previous result is used.
  CUINFER_CONNECTION_ADD = 1,    ///< Add two results.
  CUINFER_CONNECTION_MUL = 2,    ///< Multiply two results
  CUINFER_CONNECTION_CONCAT = 3, ///< Stack two results.
} cuinferTensorConnectionMode_t;

/// Profile result of convolution forward algorithms.
typedef struct {
  cuinferConvolutionFwdAlgo_t algo; ///< Algo name.
  cuinferStatus_t status;           ///< Return status.
  float time;                       ///< Runtime.
  size_t memory;                    ///< Memory needed.
  cuinferDeterminism_t determinism; ///< Is algorithm deterministic.
  cuinferMathType_t mathType;       ///< Algo math type(use tensor op or not).
  int reserved[3];                  ///< Reserved.
} cuinferConvolutionFwdAlgoPerf_t;

/// @brief Get count of convolution forward algorithms.
/// @param[in] handle The libinfer handle.
/// @param[out] count The number of convolution forward algorithms.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If \p handle is null.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionForwardAlgorithmMaxCount(
    cuinferHandle_t handle, int *count);

/// @brief Find the best convolution forward algorithm under given conditions.
/// @details Formular is given based on the combination of params.
/// * y = activate(connect((conv(x, w) * (perchannelAlpha[i] or alpha) +
/// bias[i]), z * z_scale) * alpha2)
/// * y = connect(activate(conv(x, w) * (perchannelAlpha[i] or alpha) +
/// bias[i]), z * z_scale) * alpha2
/// @todo For debug set enviroment variable \p DNN_DEBUG_FIND_CONV_FWD_ALGO to
/// the algo choosen.
/// @see ::cuinferQDEConvolutionForward
/// @param[in] handle The libinfer handle.
/// @param[in] alpha The scale factor used after convolution result. Single
/// float.
/// @param[in] perchannelAlpha The scale factor used after convolution result.
/// Channel times float.
/// @param[in] xDesc The info of input tensor x.
/// @param[in] wDesc The info of filter w.
/// @param[in] convDesc The info of convolution.
/// @param[in] yDesc The info of output tensor y.
/// @param[in] zDesc The info of input tensor z.
/// @param[in] biasDesc Not used.
/// @param[in] activationDesc The info of activation.
/// @param[in] connectionMode The connection mode.
/// @param[in] perChannel Whether alpha is individual for each channel.
/// @param[in] connectionBeforeActivation Whether activation is performed before
/// connection.
/// @param[in] requestedAlgoCount Requested algorithm max count.
/// @param[out] returnedAlgoCount Result algorithm count.
/// @param[out] perfResults Profile results.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If algo is not supported.
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI cuinferFindConvolutionForwardAlgorithm(
    cuinferHandle_t handle, const void *alpha, const void *perchannelAlpha,
    const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc,
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t yDesc,
    const cuinferTensorDescriptor_t zDesc,
    const cuinferTensorDescriptor_t biasDesc,
    const cuinferActivationDescriptor_t activationDesc,
    const cuinferTensorConnectionMode_t connectionMode, bool perChannel,
    bool connectionBeforeActivation, const int requestedAlgoCount,
    int returnedAlgoCount[], cuinferConvolutionFwdAlgoPerf_t perfResults[]);

/// @brief Find best convolution forward algorithms for \p float16.
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The descriptor of tensor x.
/// @param[in] wDesc The descriptor of filter w.
/// @param[in] convDesc The descriptor of convolution.
/// @param[in] yDesc The descriptor of tensor y.
/// @param[in] zDesc The descriptor of tensor z.
/// @param[in] biasDesc The discriptor of bias.
/// @param[in] activationDesc The discriptor of activation.
/// @param[in] connectionMode The connection mode.
/// @param[in] connectionBeforeActivation Whether activation is performed before
/// connection.
/// @param[in] requestedAlgoCount Requested algorithm max count.
/// @param[out] returnedAlgoCount Result algorithm count.
/// @param[out] perfResults Profile results.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI cuinferFindConvolutionForwardAlgorithmFP16(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc,
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t yDesc,
    const cuinferTensorDescriptor_t zDesc,
    const cuinferTensorDescriptor_t biasDesc,
    const cuinferActivationDescriptor_t activationDesc,
    const cuinferTensorConnectionMode_t connectionMode,
    bool connectionBeforeActivation, const int requestedAlgoCount,
    int returnedAlgoCount[], cuinferConvolutionFwdAlgoPerf_t perfResults[]);

/// @brief Find best convolution forward algorithm within limited workspace size
/// with actual profile.
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in] y
/// @param[in] requestedAlgoCount Requested algorithm max count.
/// @param[out] returnedAlgoCount Result algorithm count.
/// @param[out] perfResults Profile results.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes The workspace size pre-allocated.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
cuinferStatus_t CUINFERWINAPI cuinferFindConvolutionForwardAlgorithmEx(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const void *x, const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cuinferConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes);

/// @brief Find best convolution forward algorithm within limited workspace size
/// with no actual run.
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in] preference The algo preference.
/// @param[in] memoryLimitInBytes The memory limit.
/// @param[out] algo The result algorithm.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionForwardAlgorithm(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc,
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t yDesc,
    cuinferConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cuinferConvolutionFwdAlgo_t *algo);

/// @brief Find the best convolution forward algorithm.
/// @param[in] handle The libinfer handle.
/// @param[in] srcDesc The discriptor of input tensor.
/// @param[in] filterDesc The discriptor of filter tensor.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] destDesc The discriptor of output tensor.
/// @param[in] requestedAlgoCount Requested algorithm max count.
/// @param[out] returnedAlgoCount Result algorithm count.
/// @param[out] perfResults Profile results.
/// @return
/// * ::CUINFER_STATUS_SUCCESS If success.
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
/// @todo This is not used by any other library.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionForwardAlgorithm_v7(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t srcDesc,
    const cuinferFilterDescriptor_t filterDesc,
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cuinferConvolutionFwdAlgoPerf_t *perfResults);

/// @brief Get extra workspace size in bytes used by convolution forward
/// algorithm.
/// @details Convolution algorithm (which requires potentially some workspace).
/// Helper function to return the minimum size of the workspace to be passed to
/// the convolution given an algo.
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in] algo The algorithm specified.
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If algo not supported.
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI cuinferGetConvolutionForwardWorkspaceSize(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc,
    const cuinferConvolutionDescriptor_t convDesc,
    const cuinferTensorDescriptor_t yDesc, cuinferConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes);

// clang-format off
/// @defgroup ConvolutionFunctions Convollution Functions
/// @details 
/// Common result for all convolution functions for quantifier.
/// \code
/// if cuinferTensorConnectionMode_t == CUINFER_CONNECTION_NONE:
///     if bias == nullptr:
///         if perChannel == false:
///             y = clip(round(activate((convtransposed(x, w) * alpha))))
///         if perChannel == ture:
///             y = clip(round(activate((convtransposed(x, w) * perchannelAlpha[i]))))
///     if bias != nullptr:
///         if perChannel == false:
///             y = clip(round(activate((convtransposed(x, w) * alpha) + bias[0])))
///         if perChannel == ture:
///             y = clip(round(activate((convtransposed(x, w) * perchannelAlpha[i]) + bias[i])))
/// elif cuinferTensorConnectionMode_t == CUINFER_CONNECTION_ADD:
///     if bias == nullptr:
///         if perChannel == false:
///             if befor_activation_ == false:
///                 y = clip(round((activate(convtransposed(x, w) * alpha) + z * z_scale) * alpha2))
///             else:
///                 y = clip(round(activate(((convtransposed(x, w) * alpha) + z * z_scale) * alpha2)))
///         if perChannel == ture:
///             if connectionDesc.befor_activation_ == 0:
///                 y = clip(round((activate(convtransposed(x, w) * perchannelAlpha[i]) + z * z_scale) * alpha2))
///             else:
///                 y = clip(round(activate(((convtransposed(x, w) * perchannelAlpha[i]) + z * z_scale) * alpha2)))
///     if bias != nullptr:
///         if perChannel == false:
///             if befor_activation_ == false:
///                 y = clip(round((activate(convtransposed(x, w) * alpha + bias[i]) + z * z_scale) * alpha2))
///             else:
///                 y = clip(round(activate(((convtransposed(x, w) * alpha + bias[i]) + z * z_scale) * alpha2)))
///         if perChannel == ture:
///             if befor_activation_ == false:
///                 y = clip(round((activate(convtransposed(x, w) * perchannelAlpha[i] + bias[i]) + z * z_scale) * alpha2))
///             else:
///                 y = clip(round(activate(((convtransposed(x, w) * perchannelAlpha[i] + bias[i]) + z * z_scale) * alpha2)))
/// elif cuinferTensorConnectionMode_t == CUINFER_CONNECTION_CONCAT:
///     if bias == nullptr:
///         if perChannel == false:
///             if befor_activation_ == false:
///                 y = clip(round((concat(activate(convtransposed(x, w) * alpha), z * z_scale)) * alpha2))
///             else:
///                 y = clip(round(activate(concat(convtransposed(x, w) * alpha, z * z_scale)) * alpha2))
///         if perChannel == ture:
///             if befor_activation_ == false:
///                 y = clip(round((concat(activate(convtransposed(x, w) * perchannelAlpha[i]), z * z_scale)) * alpha2))
///             else:
///                 y = clip(round(activate(concat(convtransposed(x, w) * perchannelAlpha[i], z * z_scale) * alpha2)))
///     if bias != nullptr:
///         if perChannel == false:
///             if befor_activation_ == false:
///                 y = clip(round(concat(activate(convtransposed(x, w) * alpha + bias[i]), z * z_scale) * alpha2))
///             else:
///                 y = clip(round(activate(concat(convtransposed(x, w) * alpha + bias[i], z * z_scale)) * alpha2))
///         if perChannel == ture:
///             if befor_activation_ == false:
///                 y = clip(round(concat(activate(convtransposed(x, w) * perchannelAlpha[i] + bias[i]), z * z_scale) * alpha2))
///             else:
///                 y = clip(round(activate(concat((convtransposed(x, w) * perchannelAlpha[i] + bias[i]), z * z_scale) * alpha2)))
/// \endcode
// clang-format on

/// @brief Function to perform the forward pass for batch convolution.
/// @details Formula, y = alpha[0] * conv(x, w) + beta[0] * y.
/// @note Only int8(only relu) and true half configs are supported.
/// @ingroup ConvolutionFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] alpha The pointer to scaling factor.
/// @param[in] xDesc The descriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding
/// get workspace size helper function.
/// @param[in] workSpaceSizeInBytes The workspace size in bytes.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in, out] y The discriptor of output tensor y.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If input tensor is null or bad param.
/// * ::CUINFER_STATUS_NOT_SUPPORTED If algo not supported.
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI
cuinferConvolutionForward(cuinferHandle_t handle, const void *alpha,
                          const cuinferTensorDescriptor_t xDesc, const void *x,
                          const cuinferFilterDescriptor_t wDesc, const void *w,
                          const cuinferConvolutionDescriptor_t convDesc,
                          cuinferConvolutionFwdAlgo_t algo, void *workSpace,
                          size_t workSpaceSizeInBytes, const void *beta,
                          const cuinferTensorDescriptor_t yDesc, void *y);

/// @brief Convolution forward with quantifier.
/// @details For what params means like \p alpha, \p alpha2, and \p bias, see
/// \ref ConvolutionFunctions.
/// @ingroup ConvolutionFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] gamma Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes The workspace size in bytes.
/// @param[in] alpha2 The pointer to scaling factor
/// @param[in] zDesc
/// @param[in] z
/// @param[in] biasDesc
/// @param[in] bias
/// @param[in] activationDesc
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferQConvolutionForward(
    cuinferHandle_t handle, const void *alpha, const void *beta,
    const void *gamma, const cuinferTensorDescriptor_t xDesc, const void *x,
    const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferConvolutionDescriptor_t convDesc,
    cuinferConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *alpha2,
    const cuinferTensorDescriptor_t zDesc, const void *z,
    const cuinferTensorDescriptor_t biasDesc, const void *bias,
    const cuinferActivationDescriptor_t activationDesc,
    const cuinferTensorDescriptor_t yDesc, void *y);

/// @brief
/// @ingroup ConvolutionFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] perchannelAlpha
/// @param[in] beta Pointer to scaling factor.
/// @param[in] gamma Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @param[in] alpha2
/// @param[in] zDesc
/// @param[in] z
/// @param[in] biasDesc
/// @param[in] bias
/// @param[in] quadDesc
/// @param[in] perChannel
/// @param[in] activationDesc
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferQDConvolutionForward(
    cuinferHandle_t handle, const void *alpha, const void *perchannelAlpha,
    const void *beta, const void *gamma, const cuinferTensorDescriptor_t xDesc,
    const void *x, const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferConvolutionDescriptor_t convDesc,
    cuinferConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *alpha2,
    const cuinferTensorDescriptor_t zDesc, const void *z,
    const cuinferTensorDescriptor_t biasDesc, const void *bias,
    const cuinferTensorDescriptor_t quadDesc, bool perChannel,
    const cuinferActivationDescriptor_t activationDesc,
    const cuinferTensorDescriptor_t yDesc, void *y);

/// @brief
/// @ingroup ConvolutionFunctions
/// @see Common format in group \ref ConvolutionFunctions.
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] perchannelAlpha
/// @param[in] beta Pointer to scaling factor.
/// @param[in] gamma Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @param[in] alpha2
/// @param[in] zScale
/// @param[in] zDesc
/// @param[in] z
/// @param[in] biasDesc
/// @param[in] bias
/// @param[in] perChannel
/// @param[in] activationDesc
/// @param[in] connectionBeforeActivation Whether activation is performed before
/// connection.
/// @param[in] connectionMode The connection mode.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferQDEConvolutionForward(
    cuinferHandle_t handle, const void *alpha, const void *perchannelAlpha,
    const void *beta, const void *gamma, const cuinferTensorDescriptor_t xDesc,
    const void *x, const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferConvolutionDescriptor_t convDesc,
    cuinferConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *alpha2, const void *zScale,
    const cuinferTensorDescriptor_t zDesc, const void *z,
    const cuinferTensorDescriptor_t biasDesc, const void *bias, bool perChannel,
    const cuinferActivationDescriptor_t activationDesc,
    bool connectionBeforeActivation,
    const cuinferTensorConnectionMode_t connectionMode,
    const cuinferTensorDescriptor_t yDesc, void *y);

/// @brief
/// @ingroup ConvolutionFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] gamma Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @param[in] alpha2
/// @param[in] zDesc
/// @param[in] z
/// @param[in] biasDesc
/// @param[in] bias
/// @param[in] activationDesc
/// @param[in] connectionBeforeActivation Whether activation is performed before
/// connection.
/// @param[in] connectionMode The connection mode.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferHalfConvolution2dForward(
    cuinferHandle_t handle, const void *alpha, const void *beta,
    const void *gamma, const cuinferTensorDescriptor_t xDesc, const void *x,
    const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferConvolutionDescriptor_t convDesc,
    cuinferConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *alpha2,
    const cuinferTensorDescriptor_t zDesc, const void *z,
    const cuinferTensorDescriptor_t biasDesc, const void *bias,
    const cuinferActivationDescriptor_t activationDesc,
    bool connectionBeforeActivation,
    const cuinferTensorConnectionMode_t connectionMode,
    const cuinferTensorDescriptor_t yDesc, void *y);

typedef enum {
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_0 = 0, ///< Non-deterministic.
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,        ///< Non-deterministic.
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4, ///< Not implemented.
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
  CUINFER_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
} cuinferConvolutionBwdFilterAlgo_t;

typedef struct {
  cuinferConvolutionBwdFilterAlgo_t algo;
  cuinferStatus_t status;
  float time;
  size_t memory;
  cuinferDeterminism_t determinism;
  cuinferMathType_t mathType;
  int reserved[3];
} cuinferConvolutionBwdFilterAlgoPerf_t;

typedef enum {
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_0 = 0, ///< Non-deterministic.
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
  CUINFER_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
} cuinferConvolutionBwdDataAlgo_t;

typedef struct {
  cuinferConvolutionBwdDataAlgo_t algo;
  cuinferStatus_t status;
  float time;
  size_t memory;
  cuinferDeterminism_t determinism;
  cuinferMathType_t mathType;
  int reserved[3];
} cuinferConvolutionBwdDataAlgoPerf_t;

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[out] colBuffer
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferIm2Col(cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
              const void *x, const cuinferFilterDescriptor_t wDesc,
              const cuinferConvolutionDescriptor_t convDesc, void *colBuffer);

/// Softmax algorithm.
typedef enum {
  /// Straightforward implementation. May overflow. This is useful when
  /// input is guaranteed in range.
  CUINFER_SOFTMAX_FAST = 0,
  /// Subtract max from every point to avoid overflow.
  CUINFER_SOFTMAX_ACCURATE = 1,
  /// Add log to result. This will use algorithm accurate.
  CUINFER_SOFTMAX_LOG = 2,
} cuinferSoftmaxAlgorithm_t;

typedef enum {
  /// Compute the softmax over all C, H, W for each N.
  CUINFER_SOFTMAX_MODE_INSTANCE = 0,
  /// Compute the softmax over all C for each H, W, N.
  CUINFER_SOFTMAX_MODE_CHANNEL = 1,
  /// Compute the softmax over all W for each N, C, H.
  CUINFER_SOFTMAX_MODE_WIDTH = 2
} cuinferSoftmaxMode_t;

/// @defgroup SortmaxFunctions Softmax Funtions
/// @note Softmax functions: All of the form "output = alpha * Op(inputs) + beta
/// * output".

/// @brief Function to perform forward softmax.
/// @ingroup SortmaxFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] algo The algorithm specified.
/// @param[in] mode
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSoftmaxForward(
    cuinferHandle_t handle, cuinferSoftmaxAlgorithm_t algo,
    cuinferSoftmaxMode_t mode, const void *alpha,
    const cuinferTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cuinferTensorDescriptor_t yDesc, void *y);

/// @brief Function to perform forward dequant, softmax and quant.
/// @ingroup SoftmaxFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] algo The algorithm specified.
/// @param[in] mode
/// @param[in] quant_scale
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] zero_point
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferDeQuantSoftmaxForwardQuant(
    cuinferHandle_t handle, cuinferSoftmaxAlgorithm_t algo,
    cuinferSoftmaxMode_t mode, const void *quant_scale, ///< 2 value!
    const cuinferTensorDescriptor_t xDesc, const void *x,
    const void *zero_point, const cuinferTensorDescriptor_t yDesc, void *y);

/// Pooling mode.
typedef enum {
  CUINFER_POOLING_MAX = 0,
  CUINFER_POOLING_AVERAGE_COUNT_INCLUDE_PADDING =
      1, ///< Count for average includes padded values.
  CUINFER_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING =
      2, ///< Count for average does not include padded values.
  CUINFER_POOLING_MAX_DETERMINISTIC = 3
} cuinferPoolingMode_t;

/// @brief Create an instance of pooling descriptor.
/// @param[out] poolingDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferCreatePoolingDescriptor(cuinferPoolingDescriptor_t *poolingDesc);

/// @brief
/// @param[out] poolingDesc
/// @param[in] mode
/// @param[in] maxpoolingNanOpt
/// @param[in] windowHeight
/// @param[in] windowWidth
/// @param[in] verticalPadding
/// @param[in] horizontalPadding
/// @param[in] verticalStride
/// @param[in] horizontalStride
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetPooling2dDescriptor(
    cuinferPoolingDescriptor_t poolingDesc, cuinferPoolingMode_t mode,
    cuinferNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride);

/// @brief
/// @param[in] poolingDesc
/// @param[out] mode
/// @param[out] maxpoolingNanOpt
/// @param[out] windowHeight
/// @param[out] windowWidth
/// @param[out] verticalPadding
/// @param[out] horizontalPadding
/// @param[out] verticalStride
/// @param[out] horizontalStride
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetPooling2dDescriptor(
    const cuinferPoolingDescriptor_t poolingDesc, cuinferPoolingMode_t *mode,
    cuinferNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride);

/// @brief
/// @param[out] poolingDesc
/// @param[in] mode
/// @param[in] maxpoolingNanOpt
/// @param[in] nbDims
/// @param[in] windowDimA
/// @param[in] paddingA
/// @param[in] strideA
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetPoolingNdDescriptor(
    cuinferPoolingDescriptor_t poolingDesc, const cuinferPoolingMode_t mode,
    const cuinferNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[], const int strideA[]);

/// @brief
/// @param[in] poolingDesc
/// @param[in] nbDimsRequested
/// @param[out] mode
/// @param[out] maxpoolingNanOpt
/// @param[out] nbDims
/// @param[out] windowDimA
/// @param[out] paddingA
/// @param[out] strideA
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetPoolingNdDescriptor(
    const cuinferPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cuinferPoolingMode_t *mode, cuinferNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[], int paddingA[], int strideA[]);

/// @brief
/// @param[in] poolingDesc
/// @param[out] inputTensorDesc
/// @param[in] nbDims
/// @param[out] outputTensorDimA
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetPoolingNdForwardOutputDim(
    const cuinferPoolingDescriptor_t poolingDesc,
    const cuinferTensorDescriptor_t inputTensorDesc, int nbDims,
    int outputTensorDimA[]);

/// @brief
/// @param[in] poolingDesc
/// @param[in] inputTensorDesc
/// @param[out] n
/// @param[out] c
/// @param[out] h
/// @param[out] w
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetPooling2dForwardOutputDim(
    const cuinferPoolingDescriptor_t poolingDesc,
    const cuinferTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h,
    int *w);

/// @brief Destroy an instance of pooling descriptor.
/// @param[in] poolingDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDestroyPoolingDescriptor(cuinferPoolingDescriptor_t poolingDesc);

/// @defgroup PoolingFunctions Pooling Functions
/// @note Pooling functions: All of the form "output = alpha * Op(inputs) + beta
/// * output"

/// @brief Function to perform forward pooling.
/// @ingroup PoolingFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] poolingDesc
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferPoolingForward(
    cuinferHandle_t handle, const cuinferPoolingDescriptor_t poolingDesc,
    const void *alpha, const cuinferTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cuinferTensorDescriptor_t yDesc, void *y);

/// Activation Mode. @note Some activation function use extra parameters like a,
/// which can be set by ::cuinferSetActivationDescriptor.
typedef enum {
  CUINFER_ACTIVATION_SIGMOID = 0,      ///< f(x) = 1(1+e^-x).
  CUINFER_ACTIVATION_RELU = 1,         ///< f(x) = max(x, 0).
  CUINFER_ACTIVATION_TANH = 2,         ///< f(x) = tanh(x) = 2sigmod(2x)-1.
  CUINFER_ACTIVATION_CLIPPED_RELU = 3, ///< f(x) = max(min(x,ceiling),0).
  CUINFER_ACTIVATION_ELU = 4,          ///< f(x) = x if x > 0 else a(e^x-1).
  CUINFER_ACTIVATION_IDENTITY = 5,     ///< f(x) = x.
  CUINFER_ACTIVATION_LEAKY_RELU = 6,   ///< f(x) = max(x, ax). a = -0.01 i.e.
  CUINFER_ACTIVATION_SILU = 7,         ///< f(x) = x/(1 + e^-x).
  CUINFER_ACTIVATION_HARD_SWISH = 8,   ///< x*max(0,min(6,x+3))/6.
  CUINFER_ACTIVATION_HARD_SIGMOID = 9, ///< f(x) = max(0,min(1,(x+1)/2)).
  CUINFER_ACTIVATION_MISH = 10,        ///< f(x) = x*tanh(x)*log(1+e^x).
} cuinferActivationMode_t;

/// @defgroup ActivationFunctions Activation Functions
/// @note Activation functions: All of the form "output = alpha * Op(inputs) +
/// beta * output"

/// @brief
/// @ingroup ActivationFunctions
/// @param[out] activationDesc
/// @return
cuinferStatus_t CUINFERWINAPI cuinferCreateActivationDescriptor(
    cuinferActivationDescriptor_t *activationDesc);

/// @brief
/// @ingroup ActivationFunctions
/// @param[out] activationDesc
/// @param[in] mode
/// @param[in] reluNanOpt
/// @param[in] coef Ceiling for clipped RELU, alpha for ELU.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetActivationDescriptor(
    cuinferActivationDescriptor_t activationDesc, cuinferActivationMode_t mode,
    cuinferNanPropagation_t reluNanOpt, double coef);

/// @brief
/// @ingroup ActivationFunctions
/// @param[in] activationDesc
/// @param[out] mode
/// @param[out] reluNanOpt
/// @param[out] coef Ceiling for clipped RELU, alpha for ELU.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetActivationDescriptor(
    const cuinferActivationDescriptor_t activationDesc,
    cuinferActivationMode_t *mode, cuinferNanPropagation_t *reluNanOpt,
    double *coef);

/// @brief
/// @ingroup ActivationFunctions
/// @param[in] activationDesc
/// @return
cuinferStatus_t CUINFERWINAPI cuinferDestroyActivationDescriptor(
    cuinferActivationDescriptor_t activationDesc);

/// @brief Function to perform forward activation.
/// @ingroup ActivationFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] activationDesc
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferActivationForward(
    cuinferHandle_t handle, cuinferActivationDescriptor_t activationDesc,
    const void *alpha, const cuinferTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cuinferTensorDescriptor_t yDesc, void *y);

/// @defgroup LRNFunctions LRN Functions
/// @note LRN functions: output = alpha * normalize(x) + beta * old_y

/// @brief Create an instance of LRN (Local Response Normalization) descriptor.
/// @details Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults
/// from Krizhevsky'12 ImageNet paper.
/// @ingroup LRNFunctions
/// @param[out] normDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferCreateLRNDescriptor(cuinferLRNDescriptor_t *normDesc);

/// @ingroup LRNFunctions
#define CUINFER_LRN_MIN_N 1 ///< minimum allowed lrnN
/// @ingroup LRNFunctions
#define CUINFER_LRN_MAX_N 16 ///< maximum allowed lrnN
/// @ingroup LRNFunctions
#define CUINFER_LRN_MIN_K 1e-5 ///< minimum allowed lrnK
/// @ingroup LRNFunctions
#define CUINFER_LRN_MIN_BETA 0.01 ///< minimum allowed lrnBeta

/// LRN layer mode
/// @ingroup LRNFunctions
typedef enum {
  CUINFER_LRN_CROSS_CHANNEL_DIM1 =
      0, ///< Normalize across tensor's dimA[1] dimension
} cuinferLRNMode_t;

/// @brief
/// @details Uses a window [center-lookBehind, center+lookAhead], where
/// lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
/// Values of double parameters cast to tensor data type.
/// @ingroup LRNFunctions
/// @param[out] normDesc
/// @param[in] lrnN
/// @param[in] lrnAlpha
/// @param[in] lrnBeta
/// @param[in] lrnK
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferSetLRNDescriptor(cuinferLRNDescriptor_t normDesc, unsigned lrnN,
                        double lrnAlpha, double lrnBeta, double lrnK);

/// @brief Retrieve the settings currently stored in an LRN layer descriptor.
/// @details Any of the provided pointers can be NULL (no corresponding value
/// will be returned).
/// @ingroup LRNFunctions
/// @param[in] normDesc
/// @param[out] lrnN
/// @param[out] lrnAlpha
/// @param[out] lrnBeta
/// @param[out] lrnK
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferGetLRNDescriptor(cuinferLRNDescriptor_t normDesc, unsigned *lrnN,
                        double *lrnAlpha, double *lrnBeta, double *lrnK);

/// @brief Destroy an instance of LRN descriptor.
/// @ingroup LRNFunctions
/// @param[in] lrnDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDestroyLRNDescriptor(cuinferLRNDescriptor_t lrnDesc);

/// @brief LRN cross-channel forward computation.
/// @details Double parameters cast to tensor data type.
/// @ingroup LRNFunctions
/// @param[in] handle The libinfer handle.
/// @param[in] normDesc
/// @param[in] lrnMode
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] beta Pointer to scaling factor.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferLRNCrossChannelForward(
    cuinferHandle_t handle, cuinferLRNDescriptor_t normDesc,
    cuinferLRNMode_t lrnMode, const void *alpha,
    const cuinferTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cuinferTensorDescriptor_t yDesc, void *y);

typedef enum {
  /// \p bnScale, \p bnBias tensor dims are 1xCxHxWx.. (one value per
  /// CHW...-slice, normalized over N slice).
  CUINFER_BATCHNORM_PER_ACTIVATION = 0,
  /// \p bnScale, \p bnBias tensor dims are 1xCx1x1 (one value per C-dim
  /// normalized over Nx1xHxW subtensors).
  CUINFER_BATCHNORM_SPATIAL = 1,
  /// \p bnScale, \p bnBias tensor dims are 1xCx1x1 (one value per C-dim
  /// normalized over Nx1xHxW subtensors). May be faster than
  /// ::CUINFER_BATCHNORM_SPATIAL but imposes some limits on the range of
  /// values.
  CUINFER_BATCHNORM_SPATIAL_PERSISTENT = 2,
} cuinferBatchNormMode_t;

/// Minimum epsilon allowed to be used in the Batch Normalization formula.
#define CUINFER_BN_MIN_EPSILON 0.0

/// @brief
/// @details Derives a tensor descriptor from layer data descriptor for
/// BatchNormalization \p scale, \p invVariance, \p bnBias, and \p bnScale
/// tensors. Use this tensor desc for \p bnScaleBiasMeanVarDesc and \p
/// bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
/// @param[out] derivedBnDesc
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] mode
/// @return
cuinferStatus_t CUINFERWINAPI cuinferDeriveBNTensorDescriptor(
    cuinferTensorDescriptor_t derivedBnDesc,
    const cuinferTensorDescriptor_t xDesc, cuinferBatchNormMode_t mode);

typedef enum {
  CUINFER_BATCHNORM_OPS_BN = 0,            ///< Do batch normalization only.
  CUINFER_BATCHNORM_OPS_BN_ACTIVATION = 1, ///< Do batchNorm, then activation.
  CUINFER_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2,
  ///< Do batchNorm, then elemWiseAdd, then activation.
} cuinferBatchNormOps_t;

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] mode
/// @param[in] bnOps
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] zDesc
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in] bnScaleBiasMeanVarDesc
/// @param[in] activationDesc
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cuinferHandle_t handle, cuinferBatchNormMode_t mode,
    cuinferBatchNormOps_t bnOps, const cuinferTensorDescriptor_t xDesc,
    const cuinferTensorDescriptor_t zDesc,
    const cuinferTensorDescriptor_t yDesc,
    const cuinferTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cuinferActivationDescriptor_t activationDesc, size_t *sizeInBytes);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] mode
/// @param[in] bnOps
/// @param[in] activationDesc
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferGetBatchNormalizationTrainingExReserveSpaceSize(
    cuinferHandle_t handle, cuinferBatchNormMode_t mode,
    cuinferBatchNormOps_t bnOps,
    const cuinferActivationDescriptor_t activationDesc,
    const cuinferTensorDescriptor_t xDesc, size_t *sizeInBytes);

/// @brief
/// @details Computes y = BN(x). Also accumulates moving averages of mean and
/// inverse variances.
///
/// 'Gamma'(\p bnScale) and 'Beta'(\p bnBias) respectively in Ioffe and
/// Szegedy's paper's notation.
///
/// MUST use factor=1 in the very first call of a complete training cycle.
/// Use a factor=1/(1+n) at N-th call to the function to get Cumulative Moving
/// Average (CMA) behavior \f( \mathrm{CMA|[n] = (x[1]+...+x[n])/n \f) Since
/// \f{eqnarray*}{
/// \mathrm{CMA}[n+1] &=& (n*\mathrm{CMA}[n]+x[n+1])/(n+1) \\\\
/// &=& ((n+1)*\mathrm{CMA}[n]-\mathrm{CMA}[n])/(n+1) + x[n+1]/(n+1) \\\\
/// &=& \mathrm{CMA}[n]*(1-1/(n+1)) + x[n+1]*1/(n+1)
/// \f}.
///
/// Shared desc for the next 6 tensors in the argument list. \p bnScale, \p
/// bnBias, \p resultRunningMean, \p resultRunningVariance, \p resultSaveMean
/// and \p resultSaveInvVariance.
/// * Data type to be set as follows: type = (typeOf(x) == double)
/// ? double : float Dimensions for this descriptor depend on normalization mode
/// * Spatial Normalization : tensors are expected to have dims
/// 1xCx1x1 (normalization is performed across NxHxW)
/// * Per-Activation Normalization : tensors are expected to have dims of
/// 1xCxHxW (normalization is performed across N)
/// @param[in] handle The libinfer handle.
/// @param[in] mode
/// @param[in] alpha alpha[0] = result blend factor.
/// @param[in] beta beta[0] = dest layer blend factor
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x NxCxHxW
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in] y NxCxHxW
/// @param[in] bnScaleBiasMeanVarDesc
/// @param[in] bnScale
/// @param[in] bnBias
/// @param[in] exponentialAverageFactor
/// @param[out] resultRunningMean Used in Training phase only. runningMean =
/// newMean*factor + runningMean*(1-factor).
/// @param[out] resultRunningVariance Output in training mode, input in
/// inference. Is the moving average of variance[x] (factor is applied in the
/// same way as for runningMean).
/// @param[in] epsilon Has to be >= CUINFER_BN_MIN_EPSILON. Should be the same
/// in forward and backward functions.
/// @param[out] resultSaveMean Optionally save intermediate results from the
/// forward pass here - can be reused to speed up backward pass. NULL if unused
/// @param[out] resultSaveInvVariance
/// @return
cuinferStatus_t CUINFERWINAPI cuinferBatchNormalizationForwardTraining(
    cuinferHandle_t handle, cuinferBatchNormMode_t mode, const void *alpha,
    const void *beta, const cuinferTensorDescriptor_t xDesc, const void *x,
    const cuinferTensorDescriptor_t yDesc, void *y,
    const cuinferTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance);

/// Computes y = relu(BN(x) + z). Also accumulates moving averages of mean and
/// inverse variances

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] mode
/// @param[in] bnOps
/// @param[in] alpha alpha[0] = result blend factor.
/// @param[in] beta beta[0] = dest layer blend factor
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] xData
/// @param[in] zDesc
/// @param[in] zData
/// @param[in] yDesc The discriptor of tensor y.
/// @param[in] yData
/// @param[in] bnScaleBiasMeanVarDesc
/// @param[in] bnScale
/// @param[in] bnBias
/// @param[in] exponentialAverageFactor
/// @param[out] resultRunningMean
/// @param[out] resultRunningVariance
/// @param[in] epsilon Has to be >= CUINFER_BN_MIN_EPSILON. Should be the same
/// in forward and backward functions.
/// @param[out] resultSaveMean Optionally save intermediate results from the
/// forward pass here - can be reused to speed up backward pass. NULL if unused.
/// @param[out] resultSaveInvVariance
/// @param[in] activationDesc
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @param[out] reserveSpace
/// @param[out] reserveSpaceSizeInBytes
/// @return
cuinferStatus_t CUINFERWINAPI cuinferBatchNormalizationForwardTrainingEx(
    cuinferHandle_t handle, cuinferBatchNormMode_t mode,
    cuinferBatchNormOps_t bnOps, const void *alpha, const void *beta,
    const cuinferTensorDescriptor_t xDesc, const void *xData,
    const cuinferTensorDescriptor_t zDesc, const void *zData,
    const cuinferTensorDescriptor_t yDesc, void *yData,
    const cuinferTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cuinferActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);

/// @brief Performs Batch Normalization during Inference:
/// @details y[i] = bnScale[k] * (x[i] - estimatedMean[k]) / sqrt(epsilon +
/// estimatedVariance[k]) + bnBias[k] with bnScale, bnBias, runningMean,
/// runningInvVariance tensors indexed according to spatial or per-activation
/// mode. Refer to cuinferBatchNormalizationForwardTraining above for notes on
/// function arguments.
/// @param[in] handle The libinfer handle.
/// @param[in] mode
/// @param[in] alpha alpha[0] = result blend factor
/// @param[in] beta beta[0] = dest layer blend factor
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x NxCxHxW
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y NxCxHxW
/// @param[in] bnScaleBiasMeanVarDesc
/// @param[in] bnScale
/// @param[in] bnBias
/// @param[in] estimatedMean
/// @param[in] estimatedVariance
/// @param[in] epsilon
/// @return
cuinferStatus_t CUINFERWINAPI cuinferBatchNormalizationForwardInference(
    cuinferHandle_t handle, cuinferBatchNormMode_t mode, const void *alpha,
    const void *beta, const cuinferTensorDescriptor_t xDesc, const void *x,
    const cuinferTensorDescriptor_t yDesc, void *y,
    const cuinferTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon);

/// @defgroup SpatialTransformer Spatial Transform Apis
/// @note APIs for spatial transformer network

typedef struct cuinferDropoutStruct *cuinferDropoutDescriptor_t;

/// @brief
/// @param[out] dropoutDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferCreateDropoutDescriptor(cuinferDropoutDescriptor_t *dropoutDesc);

/// @brief
/// @param[in] dropoutDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDestroyDropoutDescriptor(cuinferDropoutDescriptor_t dropoutDesc);

/// @brief Helper function to determine size of the states to be passed to
/// LibinferSetDropoutDescriptor.
/// @param[in] handle The libinfer handle.
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDropoutGetStatesSize(cuinferHandle_t handle, size_t *sizeInBytes);

/// @brief helper function to determine size of the reserve space to be passed
/// to dropout forward/backward calls.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferDropoutGetReserveSpaceSize(
    cuinferTensorDescriptor_t xdesc, size_t *sizeInBytes);

/// @brief
/// @param[in] dropoutDesc
/// @param[in] handle The libinfer handle.
/// @param[in] dropout
/// @param[out] states
/// @param[in] stateSizeInBytes
/// @param[in] seed
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferSetDropoutDescriptor(cuinferDropoutDescriptor_t dropoutDesc,
                            cuinferHandle_t handle, float dropout, void *states,
                            size_t stateSizeInBytes, unsigned long long seed);

/// @brief Restores the dropout descriptor to a previously saved-off state
/// @param dropoutDesc
/// @param handle
/// @param dropout
/// @param states
/// @param stateSizeInBytes
/// @param seed
/// @return
cuinferStatus_t CUINFERWINAPI cuinferRestoreDropoutDescriptor(
    cuinferDropoutDescriptor_t dropoutDesc, cuinferHandle_t handle,
    float dropout, void *states, size_t stateSizeInBytes,
    unsigned long long seed);

/// @brief
/// @param[in] dropoutDesc
/// @param[in] handle The libinfer handle.
/// @param[out] dropout
/// @param[out] states
/// @param[out] seed
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetDropoutDescriptor(
    cuinferDropoutDescriptor_t dropoutDesc, cuinferHandle_t handle,
    float *dropout, void **states, unsigned long long *seed);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] dropoutDesc
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] reserveSpace
/// @param[in] reserveSpaceSizeInBytes
/// @return
cuinferStatus_t CUINFERWINAPI cuinferDropoutForward(
    cuinferHandle_t handle, const cuinferDropoutDescriptor_t dropoutDesc,
    const cuinferTensorDescriptor_t xDesc, const void *x,
    const cuinferTensorDescriptor_t yDesc, void *y, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);

/// @defgroup BasicRNNAPIs Basic RNN APIs

/// @ingroup BasicRNNAPIs
typedef enum {
  CUINFER_RNN_ALGO_STANDARD = 0,
  CUINFER_RNN_ALGO_PERSIST_STATIC = 1,
  CUINFER_RNN_ALGO_PERSIST_DYNAMIC = 2,
  CUINFER_RNN_ALGO_COUNT = 3,
} cuinferRNNAlgo_t;

/// @ingroup BasicRNNAPIs
typedef enum {
  CUINFER_RNN_RELU = 0, ///< Basic RNN cell type with ReLu activation.
  CUINFER_RNN_TANH = 1, ///< Basic RNN cell type with tanh activation.
  CUINFER_LSTM = 2,     ///< LSTM with no peephole connections.
  CUINFER_GRU = 3, ///< Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' +
                   ///< z * h(t-1);
} cuinferRNNMode_t;

/// @ingroup BasicRNNAPIs
typedef enum {
  CUINFER_UNIDIRECTIONAL = 0, ///< Aingle direction network.
  CUINFER_BIDIRECTIONAL = 1,  ///< Output concatination at each layer.
} cuinferDirectionMode_t;

/// @ingroup BasicRNNAPIs
typedef enum {
  CUINFER_LINEAR_INPUT =
      0, ///< Adjustable weight matrix in first layer input GEMM.
  CUINFER_SKIP_INPUT =
      1, ///< Fixed identity matrix in the first layer input GEMM.
} cuinferRNNInputMode_t;

/// @ingroup BasicRNNAPIs
struct cuinferRNNStruct;
/// @ingroup BasicRNNAPIs
typedef struct cuinferRNNStruct *cuinferRNNDescriptor_t;

/// @ingroup BasicRNNAPIs
struct cuinferPersistentRNNPlan;
/// @ingroup BasicRNNAPIs
typedef struct cuinferPersistentRNNPlan *cuinferPersistentRNNPlan_t;

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[out] rnnDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferCreateRNNDescriptor(cuinferRNNDescriptor_t *rnnDesc);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] rnnDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDestroyRNNDescriptor(cuinferRNNDescriptor_t rnnDesc);

/// @brief
/// @details \p dataType in weight descriptors and input descriptors is used to
/// describe data/parameter storage. Dropout is between RNN layers, not between
/// recurrent steps.
/// @ingroup BasicRNNAPIs
/// @param handle
/// @param rnnDesc
/// @param hiddenSize
/// @param numLayers
/// @param dropoutDesc
/// @param inputMode
/// @param direction
/// @param mode
/// @param algo
/// @param mathPrec In the RNN descriptor is determines compute math precision,
/// modified by ::cuinferMathType_t.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetRNNDescriptor(
    cuinferHandle_t handle, cuinferRNNDescriptor_t rnnDesc,
    const int hiddenSize, const int numLayers,
    cuinferDropoutDescriptor_t dropoutDesc, cuinferRNNInputMode_t inputMode,
    cuinferDirectionMode_t direction, cuinferRNNMode_t mode,
    cuinferRNNAlgo_t algo, cuinferDataType_t mathPrec);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[out] hiddenSize
/// @param[out] numLayers
/// @param[out] dropoutDesc
/// @param[out] inputMode
/// @param[out] direction
/// @param[out] mode
/// @param[out] algo
/// @param[out] mathPrec
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNDescriptor(
    cuinferHandle_t handle, cuinferRNNDescriptor_t rnnDesc, int *hiddenSize,
    int *numLayers, cuinferDropoutDescriptor_t *dropoutDesc,
    cuinferRNNInputMode_t *inputMode, cuinferDirectionMode_t *direction,
    cuinferRNNMode_t *mode, cuinferRNNAlgo_t *algo,
    cuinferDataType_t *mathPrec);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[out] rnnDesc
/// @param[in] mType
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetRNNMatrixMathType(
    cuinferRNNDescriptor_t rnnDesc, cuinferMathType_t mType);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] rnnDesc
/// @param[out] mType
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNMatrixMathType(
    cuinferRNNDescriptor_t rnnDesc, cuinferMathType_t *mType);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[out] rnnDesc
/// @param[in] recProjSize
/// @param[in] outProjSize
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetRNNProjectionLayers(
    cuinferHandle_t handle, cuinferRNNDescriptor_t rnnDesc,
    const int recProjSize, const int outProjSize);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[out] recProjSize
/// @param[out] outProjSize
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNProjectionLayers(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    int *recProjSize, int *outProjSize);

/// @brief
/// @ingroup BasicRNNAPIs
/// @note Expensive. Creates the plan for the specific settings.
/// @param[in] rnnDesc
/// @param[in] minibatch
/// @param[in] dataType
/// @param[out] plan
/// @return
cuinferStatus_t CUINFERWINAPI cuinferCreatePersistentRNNPlan(
    cuinferRNNDescriptor_t rnnDesc, const int minibatch,
    const cuinferDataType_t dataType, cuinferPersistentRNNPlan_t *plan);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] plan
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDestroyPersistentRNNPlan(cuinferPersistentRNNPlan_t plan);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] rnnDesc
/// @param[out] plan
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetPersistentRNNPlan(
    cuinferRNNDescriptor_t rnnDesc, cuinferPersistentRNNPlan_t plan);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[out] seqLength
/// @param[out] xDesc
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNTrainingReserveSize(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const int seqLength, const cuinferTensorDescriptor_t *xDesc,
    size_t *sizeInBytes);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[out] xDesc
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @param[out] dataType
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNParamsSize(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const cuinferTensorDescriptor_t xDesc, size_t *sizeInBytes,
    cuinferDataType_t dataType);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[out] pseudoLayer
/// @param[out] xDesc
/// @param[out] wDesc
/// @param[out] w
/// @param[out] linLayerID
/// @param[out] linLayerMatDesc
/// @param[out] linLayerMat
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNLinLayerMatrixParams(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cuinferFilterDescriptor_t linLayerMatDesc, void **linLayerMat);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[out] pseudoLayer
/// @param[out] xDesc
/// @param[out] wDesc
/// @param[out] w
/// @param[out] linLayerID
/// @param[out] linLayerBiasDesc
/// @param[out] linLayerBias
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetRNNLinLayerBiasParams(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cuinferFilterDescriptor_t linLayerBiasDesc, void **linLayerBias);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[in] seqLength
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] hxDesc
/// @param[in] hx
/// @param[in] cxDesc
/// @param[in] cx
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] hyDesc
/// @param[out] hy
/// @param[in] cyDesc
/// @param[out] cy
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @return
cuinferStatus_t CUINFERWINAPI cuinferRNNForwardInference(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const int seqLength, const cuinferTensorDescriptor_t *xDesc, const void *x,
    const cuinferTensorDescriptor_t hxDesc, const void *hx,
    const cuinferTensorDescriptor_t cxDesc, const void *cx,
    const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferTensorDescriptor_t *yDesc, void *y,
    const cuinferTensorDescriptor_t hyDesc, void *hy,
    const cuinferTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes);

/// @brief
/// @ingroup BasicRNNAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[in] seqLength
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] hxDesc
/// @param[in] hx
/// @param[in] cxDesc
/// @param[in] cx
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] hyDesc
/// @param[out] hy
/// @param[in] cyDesc
/// @param[out] cy
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @param[in] reserveSpace
/// @param[in] reserveSpaceSizeInBytes
/// @return
cuinferStatus_t CUINFERWINAPI cuinferRNNForwardTraining(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const int seqLength, const cuinferTensorDescriptor_t *xDesc, const void *x,
    const cuinferTensorDescriptor_t hxDesc, const void *hx,
    const cuinferTensorDescriptor_t cxDesc, const void *cx,
    const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferTensorDescriptor_t *yDesc, void *y,
    const cuinferTensorDescriptor_t hyDesc, void *hy,
    const cuinferTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);

/// CTC LOSS
typedef enum {
  CUINFER_CTC_LOSS_ALGO_DETERMINISTIC = 0,
  CUINFER_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1
} cuinferCTCLossAlgo_t;

/// Input normalization mode for loss function
typedef enum {
  CUINFER_LOSS_NORMALIZATION_NONE = 0,
  CUINFER_LOSS_NORMALIZATION_SOFTMAX = 1
} cuinferLossNormalizationMode_t;

/// CTC (Connectionist Temporal Classification) loss descriptor
/// create/destory/set/get functions
cuinferStatus_t CUINFERWINAPI
cuinferCreateCTCLossDescriptor(cuinferCTCLossDescriptor_t *ctcLossDesc);

/// @brief
/// @param[out] ctcLossDesc
/// @param[in] compType
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetCTCLossDescriptor(
    cuinferCTCLossDescriptor_t ctcLossDesc, cuinferDataType_t compType);

/// @brief
/// @param[out] ctcLossDesc
/// @param[in] compType
/// @param[in] normMode
/// @param[in] gradMode
/// @return
cuinferStatus_t CUINFERWINAPI cuinferSetCTCLossDescriptorEx(
    cuinferCTCLossDescriptor_t ctcLossDesc, cuinferDataType_t compType,
    cuinferLossNormalizationMode_t normMode, cuinferNanPropagation_t gradMode);

/// @brief
/// @param[out] ctcLossDesc
/// @param[in] compType
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetCTCLossDescriptor(
    cuinferCTCLossDescriptor_t ctcLossDesc, cuinferDataType_t *compType);

/// @brief
/// @param[out] ctcLossDesc
/// @param[in] compType
/// @param[in] normMode
/// @param[in] gradMode
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetCTCLossDescriptorEx(
    cuinferCTCLossDescriptor_t ctcLossDesc, cuinferDataType_t *compType,
    cuinferLossNormalizationMode_t *normMode,
    cuinferNanPropagation_t *gradMode);

/// @brief
/// @param[in] ctcLossDesc
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferDestroyCTCLossDescriptor(cuinferCTCLossDescriptor_t ctcLossDesc);

/// @brief Return the ctc costs and gradients, given the probabilities and
/// labels.
/// @param[in] handle The libinfer handle.
/// @param[in] probsDesc Tensor descriptor for probabilities, the dimensions are
/// T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet
/// size).
/// @param[in] probs Probabilities after softmax, in GPU memory.
/// @param[in] labels Labels, in CPU memory.
/// @param[in] labelLengths The length of each label, in CPU memory.
/// @param[in] inputLengths The lengths of timing steps in each batch, in CPU
/// memory.
/// @param[out] costs The returned costs of CTC, in GPU memory.
/// @param[in] gradientsDesc Tensor descriptor for gradients, the dimensions
/// are T,N,A.
/// @param[out] gradients The returned CTC gradients, in GPU memory, to compute
/// costs only, set it to NULL.
/// @param[in] algo Algorithm selected, supported now 0 and 1.
/// @param[in] ctcLossDesc
/// @param[in] workspace Pointer to the workspace, in GPU memory.
/// @param[in] workSpaceSizeInBytes Size of the workspace.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferCTCLoss(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t probsDesc,
    const void *probs, const int *labels, const int *labelLengths,
    const int *inputLengths, void *costs,
    const cuinferTensorDescriptor_t gradientsDesc, void *gradients,
    cuinferCTCLossAlgo_t algo, cuinferCTCLossDescriptor_t ctcLossDesc,
    void *workspace, size_t workSpaceSizeInBytes);

/// return the workspace size needed for ctc

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] probsDesc Tensor descriptor for probabilities, the dimensions are
/// T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet
/// size).
/// @param[in] gradientsDesc Tensor descriptor for gradients, the dimensions are
/// T,N,A. To compute costs only, set it to nullptr.
/// @param[in] labels labels, in CPU memory
/// @param[in] labelLengths The length of each label, in CPU memory
/// @param[in] inputLengths The lengths of timing steps in each batch, in CPU
/// memory
/// @param[in] algo The algorithm selected. Algo 0 and 1 are supported for now.
/// @param[in] ctcLossDesc
/// @param[out] sizeInBytes pointer to the returned workspace size
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetCTCLossWorkspaceSize(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t probsDesc,
    const cuinferTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths, cuinferCTCLossAlgo_t algo,
    cuinferCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes);

typedef struct {
  union Algorithm {
    cuinferConvolutionFwdAlgo_t convFwdAlgo;
    cuinferConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    cuinferConvolutionBwdDataAlgo_t convBwdDataAlgo;
    cuinferRNNAlgo_t RNNAlgo;
    cuinferCTCLossAlgo_t CTCLossAlgo;
  } algo;
} cuinferAlgorithm_t;

/// Struct containing useful informaiton for each API call.
typedef struct {
  unsigned cuinfer_version;
  cuinferStatus_t cuinferStatus;
  unsigned time_sec;      ///< Epoch time in seconds.
  unsigned time_usec;     ///< Microseconds part of epoch time.
  unsigned time_delta;    ///< time since start in seconds.
  cuinferHandle_t handle; ///< Cuinfer handle.
  cudaStream_t stream;    ///< Cuda stream ID.
  unsigned long long pid; ///< Process ID.
  unsigned long long tid; ///< Thread ID.
  int cudaDeviceId;       ///< CUDA device ID.
  int reserved[15];       ///< Reserved for future use.
} cuinferDebug_t;

/// @defgroup BertBaseInt8TransformerFunctions Bert Base Int8 Transformer
/// Functions

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] token_emb
/// @param[in] pos_emb
/// @param[in] tokens
/// @param[out] output
/// @param[out] pad_mask
/// @param[in] pad_id
/// @param[in] batch_size
/// @param[in] seq_len
/// @param[in] hidden_dim
/// @param[in] stream
/// @param[in] lang_emb
/// @param[in] lang_id
/// @param[in] multilg_type
/// @param[in] dequant_scale
/// @param[in] scaled
/// @return * CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferEncEmbI8I(const void *token_emb, const void *pos_emb,
                                 const void *tokens, void *output,
                                 void *pad_mask, int pad_id, int batch_size,
                                 int seq_len, int hidden_dim,
                                 cudaStream_t stream, const void *lang_emb,
                                 const void *lang_id, int multilg_type,
                                 float dequant_scale, bool scaled);

/// @brief
/// @details Description: from ixrt cuinferEncEmbI8I,
/// and the pad_mask is int32 instead of int8 from previous interface.
///
/// Params Mapping:
/// | src           | dst            |
/// |---------------|----------------|
/// | token_emb     | token_emb      |
/// | pos_emb       | pos_emb        |
/// | tokens        | tokens         |
/// | output        | output         |
/// | pad_mask      | pad_masktokens |
/// | pad_id        | pad_id         |
/// | batch_size    | batch_size     |
/// | seq_len       | seq_len        |
/// | hidden_dim    | hidden_dim     |
/// | stream        | stream         |
/// | lang_emb      | lang_emb       |
/// | lang_id       | lang_id        |
/// | multilg_type  | multilg_type   |
/// | dequant_scale | dequant_scale  |
/// | scaled        | scaled         |
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] token_emb
/// @param[in] pos_emb
/// @param[in] tokens
/// @param[out] output
/// @param[out] pad_mask
/// @param[in] pad_id
/// @param[in] batch_size
/// @param[in] seq_len
/// @param[in] hidden_dim
/// @param[in] stream
/// @param[in] lang_emb
/// @param[in] lang_id
/// @param[in] multilg_type
/// @param[in] dequant_scale
/// @param[in] scaled
/// @return * CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferEncEmbI8I_M8I(const void *token_emb, const void *pos_emb,
                                     const void *tokens, void *output,
                                     void *pad_mask, int pad_id, int batch_size,
                                     int seq_len, int hidden_dim,
                                     cudaStream_t stream, const void *lang_emb,
                                     const void *lang_id, int multilg_type,
                                     float dequant_scale, bool scaled);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] token_num
/// @param[in] hidden_size
/// @param[in] stream
/// @param[in, out] input
/// @param[out] output
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual_bias
/// @param[in] quant_scale
/// @param[in] is_post_ln
/// @param[in] out_col32
/// @return * CUINFER_STATUS_SUCCESS
cuinferStatus_t
cuinferLayernormResualI8O(int token_num, int hidden_size, cudaStream_t stream,
                          void *input, void *output, const void *scale,
                          const void *bias, const void *residual_bias,
                          float quant_scale, bool is_post_ln, bool out_col32);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] batch_token_num
/// @param[in] hidden_size
/// @param[in] stream
/// @param[in] ori_qkv
/// @param[in] qkv_bias
/// @param[out] new_qkv
/// @param[in] max_batch_dim
/// @param[in] batch_seq_len
/// @param[in] dim_per_head
/// @param[in] head_num
/// @param[in] quant_scale
/// @param[in] dequant_scale
/// @param[in] in_col32
/// @return * CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferArrangeEncselfQkvI8II8O(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const void *ori_qkv, const void *qkv_bias, void *new_qkv, int max_batch_dim,
    int batch_seq_len, int dim_per_head, int head_num, float quant_scale,
    float dequant_scale, bool in_col32);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] batch_size
/// @param[in] batch_seq_len
/// @param[in] head_num
/// @param[in] stream
/// @param[out] correlation
/// @param[in] src_padding_mask
/// @param[out] outputs
/// @param[in] quant_scale
/// @param[in] dequant_scale
/// @return * CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferCorrelationSoftmaxEncselfI32II8O(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    void *correlation, const void *src_padding_mask, void *outputs,
    float quant_scale, float dequant_scale);

/// @brief
/// @details Description: from ixrt IxinferCorrelationSoftmaxEncselfI8II8O
/// seperate correlation's input and output from inplace algorithm.
///
/// Params Mapping:
/// | src              | dst              |
/// |------------------|------------------|
/// | batch_size       | batch_size       |
/// | batch_seq_len    | batch_seq_len    |
/// | head_num         | head_num         |
/// | stream           | stream           |
/// | correlation      | correlation      |
/// | src_padding_mask | src_padding_mask |
/// | outputs          | correlation      |
/// | quant_scale      | quant_scale      |
/// | dequant_scale    | dequant_scale    |
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] batch_size
/// @param[in] batch_seq_len
/// @param[in] head_num
/// @param[in] stream
/// @param[out] correlation
/// @param[in] src_padding_mask
/// @param[out] outputs
/// @param[in] quant_scale
/// @param[in] dequant_scale
/// @return * CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferCorrelationSoftmaxEncselfI8II8O(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    void *correlation, const void *src_padding_mask, void *outputs,
    float quant_scale, float dequant_scale);

/// @brief
/// @details Description: from ixrt IxinferArrangeAttenOutputI8II8O
/// defalt \p max_thread_per_block to 1024.
///
/// Params Mapping:
/// | src             | dst                  |
/// |-----------------|----------------------|
/// | batch_token_num | batch_token_num      |
/// | hidden_size     | hidden_size          |
/// | stream          | stream               |
/// | ori_q           | ori_q                |
/// | new_q           | new_q                |
/// | beam_size       | beam_size            |
/// | dim_per_head    | dim_per_head         |
/// | head_num        | head_num             |
/// | 1024            | max_thread_per_block |
/// | quant_scale     | quant_scale          |
/// | dequant_scale   | dequant_scale        |
/// | out_col32       |                      |
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] batch_token_num
/// @param[in] hidden_size
/// @param[in] stream
/// @param[in] ori_q
/// @param[out] new_q
/// @param[in] beam_size
/// @param[in] dim_per_head
/// @param[in] head_num
/// @param[in] quant_scale
/// @param[in] dequant_scale
/// @param[in] out_col32
/// @return
/// * ::CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferArrangeAttenOutputI8II8O(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const void *ori_q, void *new_q, int beam_size, int dim_per_head,
    int head_num, float quant_scale, float dequant_scale, bool out_col32);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @details Description: from ixrt IxinferLnResidualI8I
///
/// Params Mapping:
/// | src           | dst           |
/// |---------------|---------------|
/// | input         | input         |
/// | scale         | scale         |
/// | bias          | bias          |
/// | residual      | residual      |
/// | output        | output        |
/// | batch_tokens  | batch_tokens  |
/// | hidden_size   | hidden_size   |
/// | dequant_scale | dequant_scale |
/// | stream        | stream        |
/// @param[in] input
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual
/// @param[out] output
/// @param[in] batch_tokens
/// @param[in] hidden_size
/// @param[in] dequant_scale
/// @param[in] stream
/// @return
/// * ::CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferResidualBiaslnI8I(const void *input, const void *scale,
                                         const void *bias, const void *residual,
                                         void *output, int batch_tokens,
                                         int hidden_size, float dequant_scale,
                                         cudaStream_t stream);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] input
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual_bias
/// @param[out] output
/// @param[out] residual
/// @param[in] batch_tokens
/// @param[in] hidden_size
/// @param[in] dequant_scale
/// @param[in] quant_scale
/// @param[in] stream
/// @param[in] is_post_ln
/// @param[in] in_col32
/// @param[in] out_col32
/// @param[in] colsum
/// @return
/// * ::CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferResidualBiasLnI8II8O(
    const void *input, const void *scale, const void *bias,
    const void *residual_bias, void *output, void *residual, int batch_tokens,
    int hidden_size, float dequant_scale, float quant_scale,
    cudaStream_t stream, bool is_post_ln, bool in_col32, bool out_col32,
    const void *colsum = nullptr);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @details Description: from ixrt IxinferResidualBiasLnI8II8O
/// residual_out is write to residual and make it inplace.
///
/// Param Mappings:
/// | src           | dst                  |
/// |---------------|----------------------|
/// | input         | input                |
/// | scale         | scale                |
/// | bias          | bias                 |
/// | residual_bias | residual_bias        |
/// | output        | output               |
/// | residual      | residual             |
/// | residual_out  | residual             |
/// | batch_tokens  | batch_tokens         |
/// | hidden_size   | hidden_size          |
/// | dequant_scale | dequant_scale        |
/// | quant_scale   | quant_scale          |
/// | 1024          | max_thread_per_block |
/// | stream        | stream               |
/// | is_post_ln    | is_post_ln           |
/// | colsum        | colsum               |
/// @param[in] input
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual_bias
/// @param[out] output
/// @param[out] residual
/// @param[out] residual_out
/// @param[in] batch_tokens
/// @param[in] hidden_size
/// @param[in] dequant_scale
/// @param[in] quant_scale
/// @param[in] stream
/// @param[in] is_post_ln
/// @param[in] colsum
/// @return
/// * ::CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferResidualBiasLnI8II8OF(
    const void *input, const void *scale, const void *bias,
    const void *residual_bias, void *output, void *residual, void *residual_out,
    int batch_tokens, int hidden_size, float dequant_scale, float quant_scale,
    cudaStream_t stream, bool is_post_ln, const void *colsum = nullptr);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @details Description: from ixrt ViterbiDecode, template is specilized
/// according to num_tags internally, slightly change in parameters' order.
///
/// Param Mappings:
/// | src               | dst               |
/// |-------------------|-------------------|
/// | stream            | stream            |
/// | batch_size        | batch_size        |
/// | seq_len           | seq_length        |
/// | num_tags          | num_tags          |
/// | emissions         | emissions         |
/// | mask              | mask              |
/// | start_transitions | start_transitions |
/// | transitions       | transitions       |
/// | end_transitions   | end_transitions   |
/// | output            | best_path         |
/// @param[in] stream
/// @param[in] batch_size
/// @param[in] seq_len
/// @param[in] num_tags
/// @param[in, out] emissions
/// @param[in, out] mask
/// @param[out] start_transitions
/// @param[out] transitions
/// @param[out] end_transitions
/// @param[out] output
/// @return
/// * ::CUINFER_STATUS_SUCCESS
cuinferStatus_t cuinferViterbiDecode(cudaStream_t stream, int batch_size,
                                     int seq_len, int num_tags, void *emissions,
                                     void *mask, void *start_transitions,
                                     void *transitions, void *end_transitions,
                                     void *output);

/// @brief
/// @details From ixrt IxinferMhaI8Launcher.
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] stream
/// @param[in] q
/// @param[in] k
/// @param[in] v
/// @param[in] mask
/// @param[out] c
/// @param[in] batch_size
/// @param[in] head_num
/// @param[in] seq_len
/// @param[in] head_dim
/// @param[in] qmax
/// @param[in] kmax
/// @param[in] vmax
/// @param[in] smax
/// @param[in] qkmax
/// @param[in] rmax
/// @return
/// * ::CUINFER_STATUS_SUCCESS
/// * ::CUINFER_STATUS_INTERNAL_ERROR
cuinferStatus_t cuinferFusedMultiHeadAttentionI8(
    cudaStream_t stream, void *q, void *k, void *v, void *mask, void *c,
    int batch_size, int head_num, int seq_len, int head_dim, float qmax,
    float kmax, float vmax, float smax, float qkmax, float rmax);

/// @brief
/// @details Description: from ixrt IxinferBiasGeluI8II8O
///
/// Params Mapping:
/// | src             | dst           |
/// |-----------------|---------------|
/// | batch_token_num | input         |
/// | stream          | stream        |
/// | input           | input         |
/// | output          | output        |
/// | bias            | bias          |
/// | feature_dim     | feature_dim   |
/// | dequant_scale   | dequant_scale |
/// | quant_scale     | quant_scale   |
/// | in_col32        |               |
/// | out_col32       |               |
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] batch_token_num
/// @param[in] stream
/// @param[in] input
/// @param[out] output
/// @param[in] bias
/// @param[in] feature_dim
/// @param[in] dequant_scale
/// @param[in] quant_scale
/// @param[in] in_col32
/// @param[in] out_col32
/// @todo remove incol32, outcol32
/// @todo input should mark as const
/// @return
/// * ::CUINFER_STATUS_SUCCESS
/// * ::CUINFER_STATUS_INTERNAL_ERROR
cuinferStatus_t cuinferBiasGeluI8II8O(int batch_token_num, cudaStream_t stream,
                                      void *input, void *output,
                                      const void *bias, int feature_dim,
                                      float dequant_scale, float quant_scale,
                                      bool in_col32, bool out_col32);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] input
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual_bias
/// @param[out] output
/// @param[out] residual
/// @param[in] batch_tokens
/// @param[in] hidden_size
/// @param[in] dequant_scale
/// @param[in] quant_scale
/// @param[in] stream
/// @param[in] is_post_ln
/// @param[in] in_col32
/// @param[in] out_col32
/// @param[in] colsum
cuinferStatus_t cuinferResidualBiaslnI32II8O(
    const void *input, const void *scale, const void *bias,
    const void *residual_bias, void *output, void *residual, int batch_tokens,
    int hidden_size, float dequant_scale, float quant_scale,
    cudaStream_t stream, bool is_post_ln, bool in_col32, bool out_col32,
    const void *colsum);

/// @brief
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] input
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual
/// @param[out] output
/// @param[in] batch_tokens
/// @param[in] hidden_size
/// @param[in] dequant_scale
/// @param[in] stream
/// @param[in] in_col32
/// @param[in] colsum
cuinferStatus_t cuinferResidualBiaslnI32I(const void *input, const void *scale,
                                          const void *bias,
                                          const void *residual, void *output,
                                          int batch_tokens, int hidden_size,
                                          float dequant_scale,
                                          cudaStream_t stream, bool in_col32,
                                          const void *colsum);

/// @brief
/// @details Description: from ixrt IxinferLnResidualI8OLauncher
/// Params Mapping:
/// | src           | dst           |
/// |---------------|---------------|
/// | token_num     | batch_tokens  |
/// | hidden_size   | hidden_size   |
/// | stream        | stream        |
/// | input         | input         |
/// | output        | output        |
/// | residual_out  | residual      |
/// | scale         | scale         |
/// | bias          | bias          |
/// | residual_bias | residual_bias |
/// | quant_scale   | quant_scale   |
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] input
/// @param[in] scale
/// @param[in] bias
/// @param[in] residual_bias
/// @param[out] output
/// @param[out] residual_out
/// @param[in] token_num
/// @param[in] hidden_size
/// @param[in] quant_scale
/// @param[in] stream
cuinferStatus_t cuinferLayernormResidualI8OFO(
    const void *input, const void *scale, const void *bias,
    const void *residual_bias, void *output, void *residual_out, int token_num,
    int hidden_size, float quant_scale, cudaStream_t stream);

/// @brief
/// @details Description: from ixrt IxinferArrangeEncselfQkvI8II8O
/// * in_col32 will be removed todo
/// * max_thread_per_block default to 1024
/// * new_qkv result split to 3 parts and output
/// Params Maping:
/// | src             | dst                  |
/// |-----------------|----------------------|
/// | batch_token_num | batch_token_num      |
/// | hidden_size     | hidden_size          |
/// | stream          | stream               |
/// | ori_qkv         | ori_qkv              |
/// | qkv_bias        | qkv_bias             |
/// | new_q           | new_qkv              |
/// | new_k           | new_qkv              |
/// | new_v           | new_qkv              |
/// | max_batch_dim   | max_batch_dim        |
/// | batch_seq_len   | batch_seq_len        |
/// | dim_per_head    | dim_per_head         |
/// | head_num        | head_num             |
/// | 1024            | max_thread_per_block |
/// | quant_scale     | quant_scale          |
/// | dequant_scale   | dequant_scale        |
/// @ingroup BertBaseInt8TransformerFunctions
/// @param[in] batch_token_num
/// @param[in] hidden_size
/// @param[in] stream
/// @param[in] ori_qkv
/// @param[in] qkv_bias
/// @param[out] new_q
/// @param[out] new_k
/// @param[out] new_v
/// @param[in] max_batch_dim
/// @param[in] batch_seq_len
/// @param[in] dim_per_head
/// @param[in] head_num
/// @param[in] quant_scale
/// @param[in] dequant_scale
cuinferStatus_t cuinferArrangeEncselfQkvSepI8II8O(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const void *ori_qkv, const void *qkv_bias, void *new_q, void *new_k,
    void *new_v, int max_batch_dim, int batch_seq_len, int dim_per_head,
    int head_num, float quant_scale, float dequant_scale);

/// @defgroup GEMM

/// @ingroup GEMM
typedef enum {
  CUINFER_OP_N = 0,
  CUINFER_OP_T = 1,
  CUINFER_OP_C = 2,
  CUINFER_OP_ROW2_COL16_4R2 = 3,
} cuinferOperation_t;

/// @ingroup GEMM
typedef enum {
  CUINFER_POINTER_MODE_HOST,   ///< The pointer is host pointer.
  CUINFER_POINTER_MODE_DEVICE, ///< The pointer is device pointer.
} cuinferPointerMode_t;

/// @ingroup GEMM
typedef enum {
  CUINFER_BLAS_GEMM_CUSTOM_NONE = 0,
  CUINFER_BLAS_GEMM_CUSTOM_BIAS_ADD_ROW_OUT = 1,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS = 2,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU = 3,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU = 4,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_TRANSPOSE = 5,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS = 6,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_GELU = 7,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_RELU = 8,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_TRANSPOSE = 9,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_SIGMOID = 10,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_SIGMOID = 11,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_SILU = 12,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_SILU = 13,
  CUINFER_BLAS_GEMM_CUSTOM_SIGMOID = 14,
  CUINFER_BLAS_GEMM_CUSTOM_SILU = 15,
  CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_TANH = 16,
  CUINFER_BLAS_GEMM_CUSTOM_FLOATBIAS_TANH = 17,
  CUINFER_BLAS_GEMM_SPECIAL_INT8_FLOATBIAS = 18,
  CUINFER_BLAS_GEMM_SPECIAL_INT8_FLOATBIAS_GELU = 19
} cuinferGEMMCustomOption_t;

/// @brief
/// @ingroup GEMM
/// @param[in] handle The libinfer handle.
/// @param[in] stream
/// @param[in] ptrMode
/// @param[in] transa
/// @param[in] transb
/// @param[in] m
/// @param[in] n
/// @param[in] k
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] A
/// @param[in] Atype
/// @param[in] lda
/// @param[in] strideA
/// @param[in] B
/// @param[in] Btype
/// @param[in] ldb
/// @param[in] strideB
/// @param[in] beta Pointer to scaling factor.
/// @param[out] C
/// @param[in] Ctype
/// @param[in] ldc
/// @param[in] strideC
/// @param[in] batchCount
/// @param[in] computeType
/// @param[in] scaleType
/// @param[in] customHostPtr
/// @param[in] customDevicePtr
/// @param[in] customOption
/// @return
cuinferStatus_t CUINFERWINAPI cuinferCustomGemm(
    cuinferHandle_t handle, cudaStream_t stream, cuinferPointerMode_t ptrMode,
    cuinferOperation_t transa, cuinferOperation_t transb, int m, int n, int k,
    const void *alpha, const void *A, cudaDataType_t Atype, int lda,
    long long int strideA, const void *B, cudaDataType_t Btype, int ldb,
    long long int strideB, const void *beta, void *C, cudaDataType_t Ctype,
    int ldc, long long int strideC, int batchCount, cudaDataType_t computeType,
    cudaDataType_t scaleType, const void *customHostPtr,
    const void *customDevicePtr, cuinferGEMMCustomOption_t customOption);

/// @brief
/// @ingroup GEMM
/// @param[in] m
/// @param[in] n
/// @param[in] k
/// @param[in] transA
/// @param[in] transB
/// @param[in] Atype
/// @param[in] Btype
/// @param[in] Ctype
/// @param[in] computeType
/// @param[in] scaleType
/// @param[out] workspaceSize
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetCustomGemmExWorkspace(
    int m, int n, int k, cuinferOperation_t transA, cuinferOperation_t transB,
    cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
    cudaDataType_t computeType, cudaDataType_t scaleType,
    size_t *workspaceSize);

/// @brief
/// @ingroup GEMM
/// @param[in] handle The libinfer handle.
/// @param[in] stream
/// @param[in] ptrMode
/// @param[in] transa
/// @param[in] transb
/// @param[in] m
/// @param[in] n
/// @param[in] k
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] A
/// @param[in] Atype
/// @param[in] lda
/// @param[in] strideA
/// @param[in] B
/// @param[in] Btype
/// @param[in] ldb
/// @param[in] strideB
/// @param[in] beta Pointer to scaling factor.
/// @param[out] C
/// @param[in] Ctype
/// @param[in] ldc
/// @param[in] strideC
/// @param[in] batchCount
/// @param[in] computeType
/// @param[in] scaleType
/// @param[in] customHostPtr
/// @param[in] customDevicePtr
/// @param[in] customOption
/// @param[in] workspace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferCustomGemmEx(
    cuinferHandle_t handle, cudaStream_t stream, cuinferPointerMode_t ptrMode,
    cuinferOperation_t transa, cuinferOperation_t transb, int m, int n, int k,
    const void *alpha, const void *A, cudaDataType_t Atype, int lda,
    long long int strideA, const void *B, cudaDataType_t Btype, int ldb,
    long long int strideB, const void *beta, void *C, cudaDataType_t Ctype,
    int ldc, long long int strideC, int batchCount, cudaDataType_t computeType,
    cudaDataType_t scaleType, const void *customHostPtr,
    const void *customDevicePtr, cuinferGEMMCustomOption_t customOption,
    void *workspace);

/// @defgroup NMS NoN-Max Suppression(NMS)
/// @note The bounding boxex is of form [xmin, ymin, xmax, ymax, class_id,
/// score], which is 6 floats. The bounding box can be either form of pixel or
/// scaled to 0.0-1.0.

/// @brief Gpu version of Non-Max Suppression(NMS) over bounding boxex.
/// @ingroup NMS
/// @note The bounding boxex is of form [xmin, ymin, xmax, ymax, class_id,
/// score], which is 6 floats. The bounding box can be either form of pixel or
/// scaled to 0.0-1.0.
/// @param[in] handle The libinfer handle.
/// @param[in] pDetections The input bounding boxex. Device pointer. Size
/// pDetections[nInputs][6].
/// @param[in] nInputs The number of input bounding boxex.
/// @param[out] pKeepDetections The result bounding boxex. Device pointer.
/// @param[in] nMaxKeep The max result bounding boxex. 0 <= \p nKeep <= \p
/// nMaxKeep.
/// @param[out] nKeep The number of result bounding boxex to kept.
/// @param[in] fIoUThresh The IoU threshold. The bounding boxex will be
/// suppressed if iou score is over this threshold.
/// @param[in] fScoreThresh The score threshold, only higher score are come into
/// consideration.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[out] outputIndice The index of corresponding result. Set to \p
/// nullptr will disable it.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If param is invalid(mostly nMaxKeep too large).
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI
cuinferNMS(cuinferHandle_t handle, float *pDetections, const int nInputs,
           float *pKeepDetections, const int nMaxKeep, int *nKeep,
           const float fIoUThresh, const float fScoreThresh, void *workspace,
           int *outputIndice = nullptr);

/// @brief Get the workspace of the corresponding ::cuinferNMS.
/// @ingroup NMS
/// @param pDetections Not used.
/// @param[in] nInputs The number of input bounding boxex.
/// @param pKeepDetections Not used.
/// @param[in] nMaxKeep The max result bounding boxex. 0 <= \p nKeep <= \p
/// nMaxKeep.
/// @param nKeep not used.
/// @param[in] fIoUThresh The score threshold, only higher score are come into
/// consideration.
/// @param[in] fScoreThresh The score threshold, only higher score are come into
/// consideration.
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @param[in] outputIndice Whether output index of corresponding result.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If param is invalid(mostly nMaxKeep too large).
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI cuinferGetNMSWorkspaceSize(
    float *pDetections, const int nInputs, float *pKeepDetections,
    const int nMaxKeep, int *nKeep, const float fIoUThresh,
    const float fScoreThresh, size_t *sizeInBytes, bool outputIndice = false);

/// @brief The batched version of ::cuinferNMS.
/// @ingroup NMS
/// @param[in] handle The libinfer handle.
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param[in] pDetections The input bounding boxex. Device pointer. Size
/// pDetections[batch][nInputs][6].
/// @param[in] nInputs The number of input bounding boxex in each batch.
/// @param[out] pKeepDetections The result bounding boxex. Device pointer. Note
/// the padding when first fewer batchs not full.
/// @param[in] nMaxKeep The max result bounding boxex. 0 <= \p nKeep <= \p
/// nMaxKeep for every batch.
/// @param[out] nKeep The number of result bounding boxex to kept for each
/// batch. Size batch.
/// @param[in] fIoUThresh The score threshold, only higher score are come into
/// consideration.
/// @param[in] fScoreThresh The score threshold, only higher score are come into
/// consideration.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.Whether output index of corresponding
/// result.t *pKeepDetections, const int nMaxKeep,
/// * ::CUINFER_STATUS_BAD_PARAM If param is invalid(mostly nMaxKeep too large).
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI
cuinferNMSBatched(cuinferHandle_t handle, int batch, float *pDetections,
                  const int nInputs, float *pKeepDetections, const int nMaxKeep,
                  int *nKeep, const float fIoUThresh, const float fScoreThresh,
                  void *workspace, int *outputIndice = nullptr);

/// @brief Get the workspace of the corresponding ::cuinferGetNMSWorkspaceSize.
/// @ingroup NMS
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param pDetections Not used.
/// @param[in] nInputs The number of input bounding boxex in each batch.
/// @param pKeepDetections Not used.
/// @param[in] nMaxKeep The max result bounding boxex. 0 <= \p nKeep <= \p
/// nMaxKeep for every batch.
/// @param[in] nKeep The number of result bounding boxex to kept for each
/// batch. Size batch.
/// @param[in] fIoUThresh The score threshold, only higher score are come into
/// consideration.
/// @param[in] fScoreThresh The score threshold, only higher score are come into
/// consideration.
/// @param[out] sizeInBytes The result extra temporary space size in bytes.
/// @param[in] outputIndice Whether output index of corresponding result.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If param is invalid(mostly nMaxKeep too large).
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI cuinferGetNMSBatchedWorkspaceSize(
    int batch, float *pDetections, const int nInputs, float *pKeepDetections,
    const int nMaxKeep, int *nKeep, const float fIoUThresh,
    const float fScoreThresh, size_t *sizeInBytes, bool outputIndice = false);

/// @brief NMS algo specilized for Yolo format.
/// @note The output format is [x, y, w, h, boxscore, class_score1, ..., ]
/// @note Due to the nms process. Only the boxscoore with highest class_score
/// will be kept. And all other classes will be supressed.
/// @ingroup NMS
/// @param[in] handle The libinfer handle.
/// @param[in] n_batch The number of batch.
/// @param[in] n_bbox the number of bbox.
/// @param[in] detection The pointer of input tensor, size is
/// [n_batch][n_bbox][n_class+5].
/// @param[in] n_class The number of class.
/// @param[out] keep_detection The result bounding boxex. Device pointer.
/// @param[in] max_keep_per_batch The max result bounding boxex. 0 <= \p
/// n_keep_each_batch[i] <= \p max_keep_per_batch.
/// @param[out] n_keep_each_batch The result bounding boxex number for each
/// batch.
/// @param[in] iou_threshold The IoU threshold. The bounding boxex will be
/// suppressed if iou score is over this threshold.
/// @param[in] score_threshold The score threshold, only higher score are come
/// into consideration.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[out] outputIndice The index of the original input. Set to \p nullptr
/// if unused.
/// @return
/// * ::CUINFER_STATUS_BAD_PARAM If param is invalid(mostly nMaxKeep too large).
/// * ::CUINFER_STATUS_SUCCESS If success.
cuinferStatus_t CUINFERWINAPI cuinferNMSBatchedYoloFused(
    cuinferHandle_t handle, int n_batch, int n_bbox, float *detection,
    int n_class, float *keep_detection, int max_keep_per_batch,
    int *n_keep_each_batch, float iou_threshold, float score_threshold,
    void *workspace, int *outputIndice = nullptr);

/// @brief Get the workspace of the ::cuinferNMSBatchedYoloFused.
/// @ingroup NMS
/// @param[in] n_batch The number of batch.
/// @param[in] n_bbox the number of bbox.
/// @param detection Not used.
/// @param[in] n_class The number of class.
/// @param keep_detection Not used.
/// @param[in] max_keep_per_batch The max result bounding boxex. 0 <= \p
/// n_keep_each_batch[i] <= \p max_keep_per_batch.
/// @param n_keep_each_batch Not used.
/// @param[in] iou_threshold The IoU threshold. The bounding boxex will be
/// suppressed if iou score is over this threshold.
/// @param[in] score_threshold The score threshold, only higher score are come
/// into consideration.
/// @param[out] workspace_size_in_bytes The result workspace size in bytes.
/// @param[in] outputIndice Whether output index of corresponding result.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetNMSBatchedYoloFusedWorkspaceSize(
    int n_batch, int n_bbox, float *detection, int n_class,
    float *keep_detection, int max_keep_per_batch, int *n_keep_each_batch,
    float iou_threshold, float score_threshold, size_t *workspace_size_in_bytes,
    bool outputIndice = false);

/// @defgroup TransformerFMHAAPIs Transformer FHMA APIS

struct cuinferFMHAParam {
  float q_amax = 0.0f;
  float k_amax = 0.0f;
  float v_amax = 0.0f;
  float r_amax = 1.0f;
  float s_max = 1.0f;
  cuinferSoftmaxAlgorithm_t softmax_algo =
      cuinferSoftmaxAlgorithm_t::CUINFER_SOFTMAX_FAST;
};

/// @brief
/// @ingroup TransformerFMHAAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] fmha_param
/// @param[in] computeType
/// @param[in] dataType
/// @param[in] maskType
/// @param[in] q_desc
/// @param[in] q_data
/// @param[in] k_desc
/// @param[in] k_data
/// @param[in] v_desc
/// @param[in] v_data
/// @param[in] mask_desc
/// @param[in] padding_mask
/// @param[in] o_desc
/// @param[out] o_data
/// @param[in] use_tcu
/// @return
cuinferStatus_t CUINFERWINAPI cuinferFMHAForward(
    cuinferHandle_t handle, cuinferFMHAParam fmha_param,
    cuinferDataType_t computeType, cuinferDataType_t dataType,
    cuinferDataType_t maskType, const cuinferTensorDescriptor_t q_desc,
    const void *q_data, const cuinferTensorDescriptor_t k_desc,
    const void *k_data, const cuinferTensorDescriptor_t v_desc,
    const void *v_data, const cuinferTensorDescriptor_t mask_desc,
    const void *padding_mask, const cuinferTensorDescriptor_t o_desc,
    void *o_data, const bool use_tcu = true);

/// @ingroup TransformerFMHAAPIs
typedef enum {
  CUINFER_FATTN_BHSD = 0,
  CUINFER_FATTN_BSHD = 1
} cuinferFlashAttnLayout_t;

/// @ingroup TransformerFMHAAPIs
struct cuinferFMHAQuantParam {
  float q_amax;
  float k_amax;
  float v_amax;
  float p_amax;
  float o_amax;
};

/// @ingroup TransformerFMHAAPIs
typedef enum {
  CUINFER_FATTN_ALIBI_MODE_SUB_KQ = 0,
  CUINFER_FATTN_ALIBI_MODE_SQRT_SUB_QK = 1,
} cuinferFlashAttnAlibiMode_t;

/// @ingroup TransformerFMHAAPIs
struct cuinferFlashAttnConfigInfo {
  cuinferFlashAttnLayout_t layout;
  cuinferFMHAQuantParam quantParam;
  bool isCausal;
  float scaling;
  int *qoSeqArray;
  int *kvSeqArray;
  int kvSeqStart;
  int kvSeqEnd;
  int kvHeadNum;
  bool isAlibi;
  cuinferFlashAttnAlibiMode_t alibiMode;
  float *slopeM;
  int qStride;
  int kStride;
  int vStride;
};

/// @brief
/// @ingroup TransformerFMHAAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] flashAttnInfo
/// @param[in] qDesc
/// @param[in] q
/// @param[in] kDesc
/// @param[in] k
/// @param[in] vDesc
/// @param[in] v
/// @param[in] maskDesc
/// @param[in] mask
/// @param[in] oDesc
/// @param[out] o
/// @return
cuinferStatus_t CUINFERWINAPI cuinferFMHAForwardEx(
    cuinferHandle_t handle, const cuinferFlashAttnConfigInfo &flashAttnInfo,
    const cuinferTensorDescriptor_t qDesc, const void *q,
    const cuinferTensorDescriptor_t kDesc, const void *k,
    const cuinferTensorDescriptor_t vDesc, const void *v,
    const cuinferTensorDescriptor_t maskDesc, const void *mask,
    const cuinferTensorDescriptor_t oDesc, void *o);

/// @ingroup TransformerFMHAAPIs
typedef enum {
    CUINFER_GPTATTEN_CONTEXT = 0,
    CUINFER_GPTATTEN_DECODE  = 1,
} cuinferGPTFlashAttnMode_t;

/// @ingroup TransformerFMHAAPIs
struct cuinferGPTFlashAttnConfigInfo {
    cuinferGPTFlashAttnMode_t attenMode;
    float scaling;
    int qHeadnum;
    int kvHeadnum;
    int maxQSeqlen;
    const int* seqArray;
};

/// @brief
/// @ingroup TransformerFMHAAPIs
/// @param[in] handle The libinfer handle.
/// @param[in] flashAttnInfo config params of tensorrt llm fmha
/// @param[in] qkvDesc The discriptor of input tensor qkv.
/// @param[in] qkv Const pointer to input tensor qkv.
/// @param[in] pastkvDesc The discriptor of input tensor kv cache.
/// @param[in] pastkv Const pointer to input tensor kv cache.
/// @param[in] oDesc The discriptor of output tensor o.
/// @param[out] o Pointer to output tensor o.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGPTFMHAForward(
    cuinferHandle_t handle,
    const cuinferGPTFlashAttnConfigInfo& flashAttnInfo,
    const cuinferTensorDescriptor_t qkvDesc,
    const void* qkv,
    const cuinferTensorDescriptor_t pastkvDesc,
    const void* pastkv,
    const cuinferTensorDescriptor_t oDesc,
    void* o);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] x_desc
/// @param[in] x Const pointer to input tensor x.
/// @param[in] y_desc
/// @param[out] y The discriptor of output tensor y.
/// @param[in] resize_method
/// @param[in] size_h
/// @param[in] size_w
/// @param[in] top
/// @param[in] left
/// @return
cuinferStatus_t CUINFERWINAPI cuinferCropAndResize(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t x_desc,
    const void *x, const cuinferTensorDescriptor_t y_desc, void *y,
    cuinferInterpolationFlag_t resize_method, int size_h, int size_w, int top,
    int left);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] x Const pointer to input tensor x.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] data_in_type
/// @param[in] compute_type
/// @param[in] data_out_type
/// @param[in] anchor_num
/// @param[in] anchors
/// @param[in] grid
/// @param[in] stride
/// @param[in] num_class
/// @param[in] n_batch
/// @param[in] anchor_first
/// @return
cuinferStatus_t CUINFERWINAPI cuinferYoloV5Detect(
    cuinferHandle_t handle, const void *x, void *y,
    cuinferDataType_t data_in_type, cuinferDataType_t compute_type,
    cuinferDataType_t data_out_type, int anchor_num, const int *anchors,
    int grid, int stride, int num_class, int n_batch, bool anchor_first);

/// @defgroup LayerNorm Layer Norm

/// @brief
/// @ingroup LayerNorm
/// @note Only serves 2-dim N and C
/// @param[in] handle The libinfer handle.
/// @param[in] x Const pointer to input tensor x.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] data_in_type
/// @param[in] compute_type
/// @param[in] data_out_type
/// @param[in] n
/// @param[in] c
/// @param[in] scale
/// @param[in] bias
/// @param[in] epsilon
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferLayerNorm(cuinferHandle_t handle, const void *x, void *y,
                 cuinferDataType_t data_in_type, cuinferDataType_t compute_type,
                 cuinferDataType_t data_out_type, int n, int c,
                 const void *scale, const void *bias, const float epsilon);

/// @brief
/// @ingroup LayerNorm
/// @param[in] handle The libinfer handle.
/// @param[in] data_type
/// @param[in] input
/// @param[in] ln_scale
/// @param[in] ln_bias
/// @param[in] residual_bias
/// @param[in] residual_in
/// @param[out] residual_out
/// @param[out] output
/// @param[in] batch_tokens
/// @param[in] hidden_size
/// @param[in] is_postln
/// @param[in] epsilon
/// @return
cuinferStatus_t CUINFERWINAPI cuinferBiasResidualLn(
    cuinferHandle_t handle, cuinferDataType_t data_type, const void *input,
    const void *ln_scale, const void *ln_bias, const void *residual_bias,
    const void *residual_in, void *residual_out, void *output, int batch_tokens,
    int hidden_size, bool is_postln, float epsilon);

/// @defgroup GroupNorm Group Norm

/// @ingroup GroupNorm
typedef enum {
  CUINFER_GROUPNORM_AFFINE_NONE = 0,
  CUINFER_GROUPNORM_AFFINE_PERCHANNEL = 1,
  CUINFER_GROUPNORM_AFFINE_PERGROUP = 2,
} cuinferGroupNormAffineMode;

/// @brief
/// @ingroup GroupNorm
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] scale
/// @param[in] bias
/// @param[in] num_groups
/// @param[in] affineMode
/// @param[in] y
/// @param[in] epsilon
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGroupNorm(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const void *x, const void *scale, const void *bias, const int num_groups,
    cuinferGroupNormAffineMode affineMode, void *y, const float epsilon);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] scale
/// @param[in] bias
/// @param[out] y The discriptor of output tensor y.
/// @param[in] epsilon
/// @return
cuinferStatus_t CUINFERWINAPI cuinferInstanceNorm(
    cuinferHandle_t handle, const cuinferTensorDescriptor_t xDesc,
    const void *x, const void *scale, const void *bias, void *y,
    const float epsilon);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] x_desc
/// @param[in] x Const pointer to input tensor x.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] n_index
/// @param[in] c_index
/// @param[in] d_index
/// @param[in] h_index
/// @param[in] w_index
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferTranspose(cuinferHandle_t handle, const cuinferTensorDescriptor_t x_desc,
                 const void *x, void *y, unsigned n_index, unsigned c_index,
                 unsigned d_index, unsigned h_index, unsigned w_index);

/// @defgroup TransposedConv Transposed Conv

/// @ingroup TransposedConv
typedef enum {
  CUINFER_CONVOLUTION_TRANSPOSE_ALGO_AUTO = 0,           ///< Recommand default.
  CUINFER_CONVOLUTION_TRANSPOSE_ALGO_DIRECT = 1,         ///< Todo.
  CUINFER_CONVOLUTION_TRANSPOSE_ALGO_EXPLICIT_GEMM = 2,  ///< For large batch.
  CUINFER_CONVOLUTION_TRANSPOSE_ALGO_EXPLICIT_GEMM2 = 3, ///< For small c.
  CUINFER_CONVOLUTION_TRANSPOSE_ALGO_IMPLICIT_GEMM = 4,  ///< Todo.
  CUINFER_CONVOLUTION_TRANSPOSE_ALGO_COUNT = 5,
} cuinferConvolutionTransposeAlgo_t;

/// @brief
/// @ingroup TransposedConv
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[out] workSpaceSizeInBytes
/// @param[in] zDesc
/// @param[in] biasDesc
/// @param[in] activationDesc
/// @param[in] connectionMode The connection mode.
/// @param[in] yDesc The discriptor of tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetQDEConvolutionTransposedWorkspaceSize(
    const cuinferTensorDescriptor_t xDesc,
    const cuinferFilterDescriptor_t wDesc,
    const cuinferConvolutionDescriptor_t convDesc,
    cuinferConvolutionTransposeAlgo_t algo, size_t *workSpaceSizeInBytes,
    const cuinferTensorDescriptor_t zDesc,
    const cuinferTensorDescriptor_t biasDesc,
    const cuinferActivationDescriptor_t activationDesc,
    cuinferTensorConnectionMode_t connectionMode,
    const cuinferTensorDescriptor_t yDesc);

/// @brief
/// @details y = clip(round(activate(alpha * conv(x, w) + z * beta + bias) *
/// alpha2)) biasDesc is not used, zDesc == yDesc
/// @ingroup TransposedConv
/// @param[in] handle The libinfer handle.
/// @param[in] alpha Pointer to scaling factor.
/// @param[in] perchannelAlpha
/// @param[in] beta Pointer to scaling factor.
/// @param[in] gamma Pointer to scaling factor.
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] convDesc The discriptor of convolution.
/// @param[in] algo The algorithm specified.
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @param[in] alpha2
/// @param[in] zScale
/// @param[in] zDesc
/// @param[in] z
/// @param[in] biasDesc
/// @param[in] bias
/// @param[in] perChannel
/// @param[in] activationDesc
/// @param[in] connectionBeforeActivation Whether activation is performed before
/// connection.
/// @param[in] connectionMode The connection mode.
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferQDEConvolutionTranspose(
    cuinferHandle_t handle, const void *alpha, const void *perchannelAlpha,
    const void *beta, const void *gamma, const cuinferTensorDescriptor_t xDesc,
    const void *x, const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferConvolutionDescriptor_t convDesc,
    cuinferConvolutionTransposeAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *alpha2, const void *zScale,
    const cuinferTensorDescriptor_t zDesc, const void *z,
    const cuinferTensorDescriptor_t biasDesc, const void *bias, bool perChannel,
    const cuinferActivationDescriptor_t activationDesc,
    bool connectionBeforeActivation,
    cuinferTensorConnectionMode_t connectionMode,
    const cuinferTensorDescriptor_t yDesc, void *y);

/// @defgroup TopK Top-K

/// @brief
/// @ingroup TopK
/// @param[in] n
/// @param[in] m
/// @param[in] top_k
/// @param[in] sort_dim
/// @param[in] largest
/// @param[in] sorted
/// @param[in] out_value
/// @param[in] out_indice
/// @param[in] data_type
/// @param[out] workspace_size
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferGetTopKWorkspace(int n, int m, int top_k, int sort_dim, bool largest,
                        bool sorted, bool out_value, bool out_indice,
                        cuinferDataType_t data_type, size_t *workspace_size);

/// @brief
/// @ingroup TopK
/// @param[in] handle The libinfer handle.
/// @param[in] input
/// @param[in] n
/// @param[in] m
/// @param[in] top_k
/// @param[in] sort_dim
/// @param[in] largest
/// @param[in] sorted
/// @param[out] out_value
/// @param[out] out_indice
/// @param[in] datatype
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferTopK(cuinferHandle_t handle, const void *input, int n, int m, int top_k,
            int sort_dim, bool largest, bool sorted, void *out_value,
            int *out_indice, cuinferDataType_t datatype, void *workspace);

/// @brief
/// @ingroup TopK
/// @param[in] top_k
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param[in] n
/// @param[in] m
/// @param[in] k
/// @param[in] largest
/// @param[in] sorted
/// @param[in] sort_dim
/// @param[in] output
/// @param[in] indice
/// @param[in] datatype
/// @param[out] workspace_size
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetTopKBatchWorkspace(
    int top_k, int batch, int n, int m, int k, bool largest, bool sorted,
    int sort_dim, bool output, bool indice, cuinferDataType_t datatype,
    size_t *workspace_size);

/// @brief
/// @ingroup TopK
/// @param[in] handle The libinfer handle.
/// @param[in] input
/// @param[in] top_k
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param[in] n
/// @param[in] m
/// @param[in] k
/// @param[in] largest
/// @param[in] sorted
/// @param[in] sort_dim
/// @param[out] output
/// @param[out] indice
/// @param[in] datatype
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @return
cuinferStatus_t CUINFERWINAPI cuinferTopKBatch(
    cuinferHandle_t handle, const void *input, int top_k, int batch, int n,
    int m, int k, bool largest, bool sorted, int sort_dim, void *output,
    int *indice, cuinferDataType_t datatype, void *workspace);

/// @defgroup Reduce

/// @brief
/// @ingroup Reduce
/// @param[in] in_type
/// @param[in] acc_type
/// @param[in] out_type
/// @param[in] reduce_op
/// @param[in] n_dims
/// @param[in] dims
/// @param[in] n_reduce_dims
/// @param[in] reduce_dim_index
/// @param[out] workspace_size
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetReduceWorkspace(
    cuinferDataType_t in_type, cuinferDataType_t acc_type,
    cuinferDataType_t out_type, cuinferReduceTensorOp_t reduce_op, int n_dims,
    const int *dims, int n_reduce_dims, const int *reduce_dim_index,
    size_t *workspace_size);

/// @brief
/// @ingroup Reduce
/// @param[in] handle The libinfer handle.
/// @param[in] in
/// @param[out] out
/// @param[in] in_type
/// @param[in] acc_type
/// @param[in] out_type
/// @param[in] reduce_op
/// @param[in] n_dims
/// @param[in] dims
/// @param[in] n_reduce_dims
/// @param[in] reduce_dim_index
/// @param[in] workspace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferReduce(cuinferHandle_t handle, const void *in, void *out,
              cuinferDataType_t in_type, cuinferDataType_t acc_type,
              cuinferDataType_t out_type, cuinferReduceTensorOp_t reduce_op,
              int n_dims, const int *dims, int n_reduce_dims,
              const int *reduce_dim_index, void *workspace);

/// @defgroup HammingDistance Hamming Distance

/// @ingroup HammingDistance
typedef enum {
  CUINFER_HAMMING_DISTANCE_MODE_PER_BIT,
  CUINFER_HAMMING_DISTANCE_MODE_PER_CHAR,
} cuinferHammingDistanceMode;

/// @brief
/// @ingroup HammingDistance
/// @param[in] n
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param[in] mode
/// @param[out] workspace_size
/// @return
cuinferStatus_t CUINFERWINAPI cuinferGetHammingDistanceWorkspace(
    int n, int batch, cuinferHammingDistanceMode mode, size_t *workspace_size);

/// @brief
/// @ingroup HammingDistance
/// @param[in] handle The libinfer handle.
/// @param[in] in_x
/// @param[in] in_y
/// @param[out] out
/// @param[in] n
/// @param[in] batch The batch. A quantity used or made at one time.
/// @param[in] mode
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @return
cuinferStatus_t CUINFERWINAPI
cuinferHammingDistance(cuinferHandle_t handle, const unsigned char *in_x,
                       const unsigned char *in_y, int *out, int n, int batch,
                       cuinferHammingDistanceMode mode, void *workspace);

/// @brief
/// @param[in] handle The libinfer handle.
/// @param[in] rnnDesc
/// @param[in] seqLength
/// @param[in] xDesc The discriptor of input tensor x.
/// @param[in] x Const pointer to input tensor x.
/// @param[in] hxDesc
/// @param[in] hx
/// @param[in] cxDesc
/// @param[in] cx
/// @param[in] wDesc The discriptor of filter w.
/// @param[in] The const pointer of input filter w.
/// @param[in] rDesc
/// @param[in] r
/// @param[in] biasDesc
/// @param[in] bias
/// @param[in] yDesc The discriptor of tensor y.
/// @param[out] y The discriptor of output tensor y.
/// @param[in] hyDesc
/// @param[out] hy
/// @param[in] cyDesc
/// @param[out] cy
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] workSpaceSizeInBytes
/// @return
cuinferStatus_t CUINFERWINAPI cuinferLSTMForwardInference(
    cuinferHandle_t handle, const cuinferRNNDescriptor_t rnnDesc,
    const int seqLength, const cuinferTensorDescriptor_t xDesc, const void *x,
    const cuinferTensorDescriptor_t hxDesc, const void *hx,
    const cuinferTensorDescriptor_t cxDesc, const void *cx,
    const cuinferFilterDescriptor_t wDesc, const void *w,
    const cuinferFilterDescriptor_t rDesc, const void *r,
    const cuinferTensorDescriptor_t biasDesc, const void *bias,
    const cuinferTensorDescriptor_t yDesc, void *y,
    const cuinferTensorDescriptor_t hyDesc, void *hy,
    const cuinferTensorDescriptor_t cyDesc, void *cy, void *workSpace,
    size_t workSpaceSizeInBytes);

/// @defgroup PageAttention Page Attension

/// @brief
/// @ingroup PageAttention
/// @param[in] num_seqs
/// @param[in] num_heads
/// @param[in] block_size
/// @param[in] max_context_len
/// @param[out] workspaceSize
/// @return
cuinferStatus_t CUINFERWINAPI cuInferPageAttentionGetWorkspaceV2(
    unsigned num_seqs, unsigned num_heads, unsigned block_size,
    unsigned max_context_len, size_t *workspaceSize);

/// @brief
/// @ingroup PageAttention
/// @param[in] num_seqs
/// @param[in] num_heads
/// @param[in] head_size
/// @param[in] block_size
/// @param[in] max_context_len
/// @param[out] workspaceSize
/// @return
cuinferStatus_t CUINFERWINAPI cuInferPageAttentionGetWorkspace(
    unsigned num_seqs, unsigned num_heads, unsigned head_size,
    unsigned block_size, unsigned max_context_len, size_t *workspaceSize);

/// @brief
/// @ingroup PageAttention
/// @param[in] handle The libinfer handle.
/// @param[out] out_ptr
/// @param[in] outType
/// @param[in] query_ptr
/// @param[in] queryType
/// @param[in] num_seqs
/// @param[in] num_heads
/// @param[in] head_size
/// @param[in] query_stride
/// @param[in] kv_block_stride
/// @param[in] kv_head_stride
/// @param[in] key_cache_ptr
/// @param[in] keyCacheType
/// @param[in] value_cache_ptr
/// @param[in] valueCacheType
/// @param[in] block_size
/// @param[in] head_mapping
/// @param[in] scale
/// @param[in] block_tables_ptr
/// @param[in] max_num_blocks_per_seq
/// @param[in] context_lens_ptr
/// @param[in] max_context_len
/// @param[in] alibi_slopes_ptr
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] alibi_sqrt
/// @return
cuinferStatus_t CUINFERWINAPI cuInferPageAttentionV2(
    cuinferHandle_t handle, void *__restrict__ out_ptr, cudaDataType_t outType,
    const void *__restrict__ query_ptr, cudaDataType_t queryType, int num_seqs,
    int num_heads, int head_size, int query_stride, int kv_block_stride,
    int kv_head_stride, const void *__restrict__ key_cache_ptr,
    cudaDataType_t keyCacheType, const void *__restrict__ value_cache_ptr,
    cudaDataType_t valueCacheType, int block_size, const int *head_mapping,
    float scale, const int *__restrict__ block_tables_ptr,
    int max_num_blocks_per_seq, const int *__restrict__ context_lens_ptr,
    int max_context_len, const float *__restrict__ alibi_slopes_ptr,
    void *workspace = nullptr, bool alibi_sqrt = false);

/// @brief
/// @ingroup PageAttention
/// @param[in] handle The libinfer handle.
/// @param[out] out_ptr
/// @param[in] outType
/// @param[in] query_ptr
/// @param[in] queryType
/// @param[in] num_seqs
/// @param[in] num_heads
/// @param[in] head_size
/// @param[in] query_stride
/// @param[in] kv_block_stride
/// @param[in] kv_head_stride
/// @param[in] key_cache_ptr
/// @param[in] keyCacheType
/// @param[in] value_cache_ptr
/// @param[in] valueCacheType
/// @param[in] block_size
/// @param[in] head_mapping
/// @param[in] scale
/// @param[in] block_tables_ptr
/// @param[in] max_num_blocks_per_seq
/// @param[in] context_lens_ptr
/// @param[in] max_context_len
/// @param[in] alibi_slopes_ptr
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] alibi_sqrt
/// @return
cuinferStatus_t CUINFERWINAPI cuInferPageAttention(
    cuinferHandle_t handle, void *__restrict__ out_ptr, cudaDataType_t outType,
    const void *__restrict__ query_ptr, cudaDataType_t queryType, int num_seqs,
    int num_heads, int head_size, int query_stride, int kv_block_stride,
    int kv_head_stride, const void *__restrict__ key_cache_ptr,
    cudaDataType_t keyCacheType, const void *__restrict__ value_cache_ptr,
    cudaDataType_t valueCacheType, int block_size, const int *head_mapping,
    float scale, const int *__restrict__ block_tables_ptr,
    int max_num_blocks_per_seq, const int *__restrict__ context_lens_ptr,
    int max_context_len, const float *__restrict__ alibi_slopes_ptr,
    void *workspace = nullptr, bool alibi_sqrt = false);

/// @brief
/// @ingroup PageAttention
/// @param[in] handle The libinfer handle.
/// @param[out] out_ptr
/// @param[in] outType
/// @param[in] query_ptr
/// @param[in] key_ptr
/// @param[in] value_ptr
/// @param[in] queryType
/// @param[in] num_seqs
/// @param[in] num_heads
/// @param[in] num_kv_heads
/// @param[in] head_size
/// @param[in] query_stride
/// @param[in] key_stride
/// @param[in] value_stride
/// @param[in] kv_block_stride
/// @param[in] kv_head_stride
/// @param[in] key_cache_ptr
/// @param[in] keyCacheType
/// @param[in] value_cache_ptr
/// @param[in] valueCacheType
/// @param[in] block_size
/// @param[in] head_mapping
/// @param[in] scale
/// @param[in] block_tables_ptr
/// @param[in] max_num_blocks_per_seq
/// @param[in] context_lens_ptr
/// @param[in] max_context_len
/// @param[in] alibi_slopes_ptr
/// @param[in] workSpace The workspace pre-allocated. See the corresponding get
/// workspace size helper function.
/// @param[in] alibi_sqrt
/// @return
cuinferStatus_t CUINFERWINAPI cuInferPageAttentionFuse(
    cuinferHandle_t handle, void *__restrict__ out_ptr, cudaDataType_t outType,
    const void *__restrict__ query_ptr, const void *__restrict__ key_ptr,
    const void *__restrict__ value_ptr, cudaDataType_t queryType, int num_seqs,
    int num_heads, int num_kv_heads, int head_size, int query_stride,
    int key_stride, int value_stride, int kv_block_stride, int kv_head_stride,
    const void *__restrict__ key_cache_ptr, cudaDataType_t keyCacheType,
    const void *__restrict__ value_cache_ptr, cudaDataType_t valueCacheType,
    int block_size, const int *head_mapping, float scale,
    const int *__restrict__ block_tables_ptr, int max_num_blocks_per_seq,
    const int *__restrict__ context_lens_ptr, int max_context_len,
    const float *__restrict__ alibi_slopes_ptr, void *workspace = nullptr,
    bool alibi_sqrt = false);

#if defined(__cplusplus)
}
#endif

#endif /* CUINFER_H_ */
#pragma GCC visibility pop
