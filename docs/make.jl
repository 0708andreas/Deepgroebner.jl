using Deepgroebner
using Documenter

makedocs(;
    modules=[Deepgroebner],
    authors="andreas <andreas@lha66.dk> and contributors",
    repo="https://github.com/0708andreas/Deepgroebner.jl/blob/{commit}{path}#L{line}",
    sitename="Deepgroebner.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://0708andreas.github.io/Deepgroebner.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/0708andreas/Deepgroebner.jl",
)
