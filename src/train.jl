using BSON

include("model.jl")

export train

function train()
    m = experiment()
    run(m)
    bson("experiment_1000eps_3_4_3.bson", m)
end
