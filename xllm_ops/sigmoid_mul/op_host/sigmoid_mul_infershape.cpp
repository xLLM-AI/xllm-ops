/*!
 * \file sigmoid_mul_infershape.cpp
 * \brief SigmoidMul shape inference: out = broadcast(x, gate).
 */
#include <vector>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// gert::Shape only exposes an initializer_list constructor; dispatch by rank.
static gert::Shape MakeShape(const std::vector<int64_t>& d)
{
    switch (d.size()) {
        case 1: return gert::Shape({d[0]});
        case 2: return gert::Shape({d[0], d[1]});
        case 3: return gert::Shape({d[0], d[1], d[2]});
        case 4: return gert::Shape({d[0], d[1], d[2], d[3]});
        case 5: return gert::Shape({d[0], d[1], d[2], d[3], d[4]});
        case 6: return gert::Shape({d[0], d[1], d[2], d[3], d[4], d[5]});
        case 7: return gert::Shape({d[0], d[1], d[2], d[3], d[4], d[5], d[6]});
        case 8: return gert::Shape({d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]});
        default: return gert::Shape();
    }
}

// Element-wise broadcast: out shape is the broadcast of x and gate.
static ge::graphStatus InferShape4SigmoidMul(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gateShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateShape);

    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const size_t maxRank = (xShape->GetDimNum() > gateShape->GetDimNum())
                               ? xShape->GetDimNum()
                               : gateShape->GetDimNum();
    std::vector<int64_t> dims(maxRank, 1);
    for (size_t i = 0; i < maxRank; i++) {
        int64_t xDim = 1;
        int64_t gDim = 1;
        if (i < xShape->GetDimNum()) {
            xDim = xShape->GetDim(xShape->GetDimNum() - 1 - i);
        }
        if (i < gateShape->GetDimNum()) {
            gDim = gateShape->GetDim(gateShape->GetDimNum() - 1 - i);
        }
        if (xDim != gDim) {
            // Fail only when neither dimension is 1 (truly non-broadcastable).
            OP_CHECK_IF(xDim != 1 && gDim != 1,
                        OP_LOGE(context, "SigmoidMul: shapes not broadcastable"),
                        return ge::GRAPH_FAILED);
        }
        dims[maxRank - 1 - i] = (xDim > gDim) ? xDim : gDim;
    }
    *outShape = MakeShape(dims);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidMul).InferShape(InferShape4SigmoidMul);

} // namespace ops
