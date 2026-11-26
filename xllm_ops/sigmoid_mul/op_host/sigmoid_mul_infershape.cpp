/*!
 * \file sigmoid_mul_infershape.cpp
 * \brief SigmoidMul shape inference: out shape equals x shape.
 *
 * Only two gate layouts are supported (matching the kernel):
 *   - element-wise : gate has exactly the same shape as x;
 *   - row broadcast: same rank, equal leading dims, gate trailing dim == 1.
 * Since both cases yield out == x, the output shape is a direct copy of x,
 * which also preserves unknown (-1) dimensions in dynamic graphs.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// -1 (unknown / dynamic) is compatible with any dimension value.
static inline bool DimCompatible(int64_t a, int64_t b)
{
    return a < 0 || b < 0 || a == b;
}

static ge::graphStatus InferShape4SigmoidMul(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gateShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateShape);
    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const size_t xRank = xShape->GetDimNum();
    const size_t gateRank = gateShape->GetDimNum();
    OP_CHECK_IF(xRank != gateRank,
                OP_LOGE(context, "SigmoidMul: x and gate rank mismatch"), return ge::GRAPH_FAILED);

    for (size_t i = 0; i + 1 < xRank; i++) {
        OP_CHECK_IF(!DimCompatible(xShape->GetDim(i), gateShape->GetDim(i)),
                    OP_LOGE(context, "SigmoidMul: unsupported gate layout (leading dims differ)"),
                    return ge::GRAPH_FAILED);
    }
    if (xRank > 0) {
        const int64_t xLast = xShape->GetDim(xRank - 1);
        const int64_t gateLast = gateShape->GetDim(xRank - 1);
        OP_CHECK_IF(!(gateLast == 1 || DimCompatible(xLast, gateLast)),
                    OP_LOGE(context, "SigmoidMul: unsupported gate layout (trailing dim must be 1 or equal)"),
                    return ge::GRAPH_FAILED);
    }

    // Supported layouts always produce out == x; copying also keeps unknown dims intact.
    *outShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidMul).InferShape(InferShape4SigmoidMul);

} // namespace ops
