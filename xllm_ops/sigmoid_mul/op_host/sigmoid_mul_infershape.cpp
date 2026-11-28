/*!
 * \file sigmoid_mul_infershape.cpp
 * \brief SigmoidMul shape inference: 2-D only; out = x * sigmoid(gate).
 *        Supported gate layouts: exact same shape as x, or [M, 1] row broadcast.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// bf16: 32B == 16 elements. The trailing axis supports 128 .. 16K, 32B aligned.
static bool IsSupportedTailDim(int64_t n)
{
    return n >= 128 && n <= 16384 && (n % 16) == 0;
}

static ge::graphStatus InferShape4SigmoidMul(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gateShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateShape);
    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_CHECK_IF(xShape->GetDimNum() != 2 || gateShape->GetDimNum() != 2,
                OP_LOGE(context, "SigmoidMul: only 2-D tensors supported"), return ge::GRAPH_FAILED);

    const int64_t M = xShape->GetDim(0);
    const int64_t N = xShape->GetDim(1);
    const int64_t gM = gateShape->GetDim(0);
    const int64_t gN = gateShape->GetDim(1);
    // Treat an unknown dim (<0, e.g. -1 in dynamic graphs) as compatible.
    auto dimEq = [](int64_t a, int64_t b) { return a < 0 || b < 0 || a == b; };
    // A known gate width must be 1 (row broadcast) or a supported element-wise
    // width; reject others even when x's trailing dim is dynamic.
    OP_CHECK_IF(gN >= 0 && gN != 1 && !IsSupportedTailDim(gN),
                OP_LOGE(context, "SigmoidMul: known gate width must be 1 or in [128, 16384] (32B aligned)"),
                return ge::GRAPH_FAILED);
    const bool sameShape = (dimEq(M, gM) && dimEq(N, gN));
    const bool rowBroadcast = (gN == 1 && dimEq(M, gM));
    OP_CHECK_IF(!(sameShape || rowBroadcast),
                OP_LOGE(context, "SigmoidMul: gate must be same shape as x or [M, 1]"),
                return ge::GRAPH_FAILED);
    // Unknown N is allowed here; tiling re-checks it at runtime.
    OP_CHECK_IF(N >= 0 && !IsSupportedTailDim(N),
                OP_LOGE(context, "SigmoidMul: tail dim N must be 32B aligned and in [128, 16384]"),
                return ge::GRAPH_FAILED);

    // Output always follows x.
    *outShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidMul).InferShape(InferShape4SigmoidMul);

} // namespace ops
