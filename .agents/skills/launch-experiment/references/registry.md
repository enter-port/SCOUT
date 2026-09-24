# SCOUT 实验登记系统参考

这份参考只在需要登记、编辑、同步或检查实验时读取。它描述服务器当前部署的 file-based registry，不依赖数据库或 npm。

## 服务器位置

- repo：`/root/workspace/baojiachun/scout`
- 数据基目录：`/root/workspace/baojiachun`
- registry CLI：`scout/scripts/experiment_registry.py`
- 管理页面：`http://127.0.0.1:8765`，服务会话：`scout-experiment-manager`
- 页面通过 SSH 转发访问：`ssh -N -L 8765:127.0.0.1:8765 -p 1022 root@106.14.2.243`

## 元数据结构

每条 `experiment.json` 至少包含：

```json
{
  "schema_version": 2,
  "uid": "不可变内部 ID",
  "id": "CAN-2026-09-25",
  "task": "CAN",
  "number": null,
  "category": "full",
  "title": "实验说明",
  "kind": "campaign",
  "started_at": "2026-09-25T12:00:00+08:00",
  "date_precision": "day",
  "date_basis": "registered_date",
  "aliases": [],
  "legacy_roots": [],
  "extra_sources": [],
  "data_path": "scout/data/CAN/full/CAN-2026-09-25",
  "expected_rounds": 6,
  "manual_status": "planned",
  "configuration": {},
  "tags": [],
  "notes": "",
  "archived": false
}
```

`number` 在同 task 同日唯一一条时为 `null`；重复时为 1、2、3……。`manual_status` 可为 `planned`、`running`、`completed`、`failed`、`paused` 或 `null`。为 `null` 时由源结果自动判断。

## CLI

```bash
cd /root/workspace/baojiachun/scout

# 创建；ID 由系统分配
python3 scripts/experiment_registry.py new \
  --task CAN --category full --date 2026-09-25 \
  --title '实验说明' --expected-rounds 6 \
  --config-json '{"seed":233,"arms":["ATY","DP"]}' \
  --tags 'orbit,baseline' --notes '启动前说明'

# 查询；参数可以是 ID、uid、旧别名或旧数据路径
python3 scripts/experiment_registry.py show CAN-2026-09-25

# 同步一条或全库；会重建 README、results.json、registry.json
python3 scripts/experiment_registry.py sync --id CAN-2026-09-25
python3 scripts/experiment_registry.py sync

# 检查元数据、同日编号、源文件、兼容软链和生成视图
python3 scripts/experiment_registry.py check
```

`new` 只登记和创建空数据目录，不启动训练。训练脚本必须由 launch skill 在参数复述和用户确认后启动。

## 页面字段和 API

页面是 CLI 的可视化入口。详情页有四个区域：基本记录、实验配置、结果与来源、修改历史。

- 实验配置是人工维护的 JSON 对象，适合记录 seed、arm、guidance、剂量、worker、pass@K、batch、GPU、base 等。
- 实际运行配置从已登记源文件读取；加入配置快照会保存文本、源路径和 SHA-256，不会修改历史运行输入。
- 页面保存使用 revision 乐观锁；出现冲突时重新打开后再编辑，不能覆盖别人的更新。
- 归档只隐藏记录，不删除数据。
- 源文件读取限制在服务器工作区，并且单文件最多 2 MB；压缩归档只按登记的成员路径读取，绝不解压到工作区。

API 仅监听服务器 loopback，写请求需要 `Content-Type: application/json` 和 `X-Scout-Request: 1`：

```text
GET  /api/experiments
GET  /api/experiments/<id-or-uid>
GET  /api/experiments/<id-or-uid>/source?path=<registered-source>
POST /api/experiments       {task,date,category,title,expected_rounds,configuration,tags,notes}
POST /api/experiments/<id>  {revision,changes:{title,configuration,tags,notes,manual_status,archived,expected_rounds}}
POST /api/sync              {id: optional}
GET  /api/jobs
```

模型通常应使用 CLI 或页面，不需要直接调用 API。若必须调用 API，保留返回的 `uid` 和 `revision`，并在写后运行 `check`。

## 记录更新时机

| 时机 | 必做动作 |
|---|---|
| 用户确认参数后、启动前 | `new`；填配置、base、脚本和资源说明 |
| round0 成功、round1 启动 | 状态 `running`；笔记写 session、GPU、启动时间 |
| 每轮 `TOTAL` 后 | `sync --id`；检查 seed/arm/round 和读取错误 |
| 中断或失败 | 保留源文件；状态 `failed`/`paused`；写原始错误和恢复点 |
| 最后一轮完成 | `sync --id`、`check`；状态 `completed`；写结论和下一步 |

## 兼容和迁移

迁移前的目录和实验名称保存在 `aliases`、`legacy_roots` 及 `_registry/migration_v2.json`。不要删除兼容软链，不要把历史目录重新登记成第二条实验。数据源 manifest 的 SHA-256 是判断迁移前后是否一致的依据。
