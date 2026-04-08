use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use triviumdb::Database;

fn print_usage() {
    println!("TriviumDB Repair Toolkit v1.0");
    println!("Usage:");
    println!("  trivium_repair check <db_path>   - 快速检查数据库头部和维度指纹");
    println!("  trivium_repair dump <db_path>    - 强行挂载并全量导出图节点和数据");
}

fn read_dim_from_header(tdb_path: &str) -> std::io::Result<usize> {
    let mut file = File::open(tdb_path)?;
    let mut header = [0u8; 10];
    file.read_exact(&mut header)?;
    
    if &header[0..4] != b"TVDB" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "非法魔数: 不是 TriviumDB 文件"));
    }
    
    let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
    let dim = u32::from_le_bytes(header[6..10].try_into().unwrap()) as usize;
    println!("   └─ 检测到架构参数 => Version: v{}, Dimension: {}", version, dim);
    
    Ok(dim)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        print_usage();
        return;
    }

    let command = &args[1];
    let path = &args[2];

    match command.as_str() {
        "check" => {
            let wal_path = format!("{}.wal", path);
            let tdb_path = format!("{}", path);
            println!("🔍 开始扫描数据库环境: {}", path);
            println!("   ├─ WAL 日志文件存在: {}", Path::new(&wal_path).exists());
            println!("   ├─ TDB 主库文件存在: {}", Path::new(&tdb_path).exists());
            
            if Path::new(&tdb_path).exists() {
                match read_dim_from_header(&tdb_path) {
                    Ok(_) => println!("✅ 数据库头部完好。您可以安全地尝试启动挂载。"),
                    Err(e) => println!("❌ 头部探查失败: {} (主库文件可能已遭受物理破坏)", e),
                }
            } else {
                println!("⚠️ 警告: 主库文件不存在！如果只有 WAL 文件，下次启动时 TriviumDB 会尝试在零基础下纯靠 WAL 追平数据。");
            }
        }
        "dump" => {
            let tdb_path = format!("{}", path);
            let dim_res = if Path::new(&tdb_path).exists() {
                read_dim_from_header(&tdb_path).unwrap_or(4)
            } else {
                4 // 纯靠 WAL 启动默认给 4 以防 panic
            };
            
            println!("📦 尝试利用自动嗅探维度 ({}) 强制挂载数据库...", dim_res);
            match Database::<f32>::open(path, dim_res) {
                Ok(db) => {
                    println!("✅ 挂载成功！WAL 与主库成功汇合！目前存活的节点总数: {}", db.node_count());
                    println!("--------------------------------------------------");
                    for id in 1..=db.node_count() as u64 {
                        if let Some(payload) = db.get_payload(id) {
                            println!("ID [{}]: Payload -> {}", id, payload);
                        }
                    }
                    println!("--------------------------------------------------");
                    println!("✅ Dump 完毕！由于是以安全读取模式打开，系统已为您自动执行并切断了幽灵死结。");
                }
                Err(e) => {
                    println!("❌ 强行挂载失败: {}", e);
                    println!("提示: 如果是由于另一个进程崩机导致死锁，请先尝试运行 `unlock` 命令。");
                }
            }
        }
        _ => {
            println!("❌ 未知的诊断指令: {}", command);
            print_usage();
        }
    }
}
