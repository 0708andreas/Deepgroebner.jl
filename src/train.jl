using BSON: @save

include("model.jl")

export train

function train4()
    experiments = [
        (GroebnerEnvParams(3 , 4 , 3 ), 1000),
        (GroebnerEnvParams(3 , 20, 10), 1000),
        (GroebnerEnvParams(10, 20, 10), 1000),
        (GroebnerEnvParams(3 , 20, 10), 1000),
        (GroebnerEnvParams(10, 20, 10), 1000),

    ]

    models = Array{Any, 1}(undef, length(experiments))

    Threads.@threads for i in 1:length(experiments)
        params, episodes = experiments[i]
        m = experiment(params, episodes)
        models[i] = m
        run(models[i])
    end

    @save "experiments.bson" experiments models
end


function train()
    m = experiment()
    run(m)
    @save "experiment_1000eps_3_4_3.bson" m
end
