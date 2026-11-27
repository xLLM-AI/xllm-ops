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
    const bool sameShape = (dimEq(M, gM) && dimEq(N, gN));
    const bool rowBroadcast = (gN == 1 && dimEq(M, gM));
    OP_CHECK_IF(!(sameShape || rowBroadcast),
                OP_LOGE(context, "SigmoidMul: gate must be same shape as x or [M, 1]"),
                return ge::GRAPH_FAILED);
    // Tail dim is 2048 (unknown N allowed here; tiling enforces 2048 at runtime).
    OP_CHECK_IF(N >= 0 && N != 2048, OP_LOGE(context, "SigmoidMul: tail dim N must be 2048"),
                return ge::GRAPH_FAILED);

    // Output always follows x.
    *outShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidMul).InferShape(InferShape4SigmoidMul);

} // namespace ops
