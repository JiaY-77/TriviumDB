const { spawn } = require('child_process');
const fs = require('fs');

const outPath = 'bench_output.txt';
// 删除已有的文件
if (fs.existsSync(outPath)) {
    fs.unlinkSync(outPath);
}

const outStream = fs.createWriteStream(outPath, { flags: 'a' });

console.log("🚀 开始运行 Benchmark，并将结果导出至 bench_output.txt...");

// 使用 child_process 执行 cargo bench
const p = spawn('cargo', ['bench', '--bench', 'bq_vs_brute', '--color=never'], {
    shell: true,
});

// Criterion 的进度条会包含大量的 \r 字符，为了让结果变直观，
// 我们可以过滤掉所有带有 \r 的行，或者只提取核心的精度和性能报告。
let buffer = '';

function processData(data) {
    const text = data.toString('utf8');
    buffer += text;
    
    // 如果累积了换行符，按行处理
    let lines = buffer.split('\n');
    buffer = lines.pop(); // 保留最后一行不完整的
    
    for (const line of lines) {
        // 如果包含回车符 \r 但没结束，且包含进度相关字符，就跳过以保证文件干净
        if (line.includes('Benchmarking') || line.includes('warming up') || line.includes('Collecting')) {
            // 让一些进度提示进文件也没关系，但过滤纯刷新行
            continue;
        }
        // 如果行里带 \r，把 \r 以前的全部删去
        const cleanedLine = line.replace(/.*\r/g, '').trim();
        if (cleanedLine) {
            outStream.write(cleanedLine + '\n');
            console.log("写入:", cleanedLine); // 终端也能看到一些核心信息
        }
    }
}

p.stdout.on('data', processData);
p.stderr.on('data', processData);

p.on('close', (code) => {
    if (buffer.trim()) {
        const cleanedLine = buffer.replace(/.*\r/g, '').trim();
        if (cleanedLine) {
            outStream.write(cleanedLine + '\n');
        }
    }
    outStream.end();
    console.log(`✅ Benchmark 执行完成！(退出码: ${code})`);
    console.log(`📄 结果已保存至: ${outPath}`);
    process.exit(code);
});
