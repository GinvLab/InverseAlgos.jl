using Documenter, InverseAlgos

makedocs(modules = [InverseAlgos],
         repo=Remotes.GitHub("GinvLab","InverseAlgos.jl"), #"https://github.com/GinvLab/InverseAlgos.jl"
         sitename="InverseAlgos.jl",
         authors = "Andrea Zunino",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "Home" => "index.md",
         ]
         )

deploydocs(
    repo="github.com/GinvLab/MCsamplers.jl.git", 
    devbranch = "main",
    deploy_config = Documenter.GitLab(),
    branch = "gh-pages"
)


