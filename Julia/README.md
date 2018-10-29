[关于project](https://stackoverflow.com/questions/36398629/change-package-directory-in-julia)

### 安装

#### julia

使用`yaourt julia`即可

#### Ijulia (kernel for jn)

##### 步骤

创建目录`/home/drdh/.julia/environments/v1.0`作为默认的环境(由于julia的环境可以分离、嵌套，这个相当于全局环境)

在`/home/drdh/.julia/config/startup.jl`中添加如下的启动指令

```julia
using Pkg
Pkg.activate("/home/drdh/.julia/environments/v1.0")
```

在任意位置打开`julia`输入`]`注意到提示环境为`v1.0`

然后安装`add IJulia`，如果后面没有出现`build`则使用`build IJulia`

##### 其他

[可能出错](https://www.jianshu.com/p/859475dc3dae)

查看内核`jupyter kernelspec list`

[在local env中使用Ijulia](https://github.com/JuliaLang/IJulia.jl/issues/750)



