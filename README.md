# mini path tracer

![1080x1080,spp:20000 30min](./img/output.png)
代码不超过 200 行

# Build

## Windows
```bash
clang++ -O3 -march=native -ffast-math -fopenmp -std=c++17 .\main.cpp -o renderer.exe
./renderer.exe
```

## Linux & Mac
```bash
clang++ -O3 -march=native -ffast-math -fopenmp -std=c++17 .\main.cpp -o renderer
./renderer
```

    