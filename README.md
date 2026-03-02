# nana-claw-rs

NanoClaw agent-runner의 Rust 포팅. 컨테이너 안에서 LLM 에이전트를 실행하고, 도구를 오케스트레이션하며, 호스트와 IPC로 통신한다.

## 기존 TypeScript 버전과 차이점

| | TypeScript (기존) | Rust (이 프로젝트) |
|---|---|---|
| LLM | Claude 전용 (Agent SDK) | Anthropic / OpenAI / OpenRouter |
| 바이너리 | ~300MB (Node.js + npm) | ~15MB (static binary) |
| 콜드 스타트 | ~3초 (tsc 재컴파일) | 즉시 |
| 메모리 | ~100MB | ~10MB |
| 오케스트레이터 | 없음 | 태스크 분할 → 병렬 실행 → 결과 병합 |

## 아키텍처

```
stdin (JSON) ─→ main.rs ─→ agent.rs ─→ provider/ (LLM API)
                  │              ↕
                  │         tools/ (Bash, Read, Write, Grep...)
                  │              ↕
                  │         orchestrator.rs (큰 태스크 분할/병합)
                  │
                  ├─→ ipc.rs (파일 기반 IPC 폴링)
                  ├─→ session.rs (대화 이력 관리)
                  └─→ stdout (결과 출력)
```

## 모듈 구조

```
src/
├── main.rs              # 진입점, stdin/stdout 프로토콜
├── types.rs             # 공유 타입 (ContainerInput/Output, Message, Tool 등)
├── agent.rs             # 에이전트 루프: LLM 호출 → 도구 실행 → 반복
├── orchestrator.rs      # 태스크 플래너 + 병렬 실행 + 결과 병합
├── session.rs           # JSONL 기반 세션 저장/복원
├── hooks.rs             # 대화 아카이빙 (컴팩션 시)
├── ipc.rs               # /workspace/ipc/ 파일 폴링
├── mcp_server.rs        # MCP stdio 서버 (7개 도구)
├── provider/
│   ├── mod.rs           # Provider trait
│   ├── anthropic.rs     # Anthropic Messages API
│   ├── openai.rs        # OpenAI Chat Completions API
│   └── openrouter.rs    # OpenRouter API
└── tools/
    ├── mod.rs           # Tool trait + ToolRegistry
    ├── bash.rs          # 셸 명령 실행
    ├── filesystem.rs    # Read, Write, Edit, Glob
    ├── grep.rs          # 정규식 파일 검색
    └── web.rs           # WebFetch, WebSearch (stub)
```

## Provider 설정

환경변수 또는 stdin secrets로 설정:

```bash
# Anthropic (기본값)
ANTHROPIC_API_KEY=sk-ant-...
NANOCLAW_MODEL=claude-sonnet-4-20250514   # 선택

# OpenAI
NANOCLAW_PROVIDER=openai
OPENAI_API_KEY=sk-...
NANOCLAW_MODEL=gpt-4o

# OpenRouter
NANOCLAW_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
NANOCLAW_MODEL=anthropic/claude-sonnet-4
```

## 오케스트레이터

`NANOCLAW_ORCHESTRATE=1` 설정 시 활성화.

큰 요청이 들어오면:
1. **Plan** — LLM이 태스크 복잡도 분석 + 서브태스크 분리
2. **Execute** — 의존성 없는 서브태스크들을 병렬 실행, 의존성 있으면 순차
3. **Merge** — concatenate / summarize / code_merge 전략으로 결과 통합

```
사용자: "이 프로젝트에 인증 시스템 추가해줘"
  ↓
Planner → { complexity: "large", subtasks: [
  { id: "design",  prompt: "인증 아키텍처 설계..." },
  { id: "backend", prompt: "API 엔드포인트 구현...", depends_on: ["design"] },
  { id: "frontend", prompt: "로그인 UI 구현...", depends_on: ["design"] },
  { id: "test",    prompt: "테스트 작성...", depends_on: ["backend", "frontend"] }
]}
  ↓
[design] → 완료 → [backend] + [frontend] 병렬 → 완료 → [test]
  ↓
Merger → 최종 통합 결과
```

## 빌드

```bash
cargo build --release
```

## 호스트 호환성

기존 NanoClaw 호스트(`src/container-runner.ts`)와 100% 호환:
- stdin: `ContainerInput` JSON
- stdout: `---NANOCLAW_OUTPUT_START/END---` 마커
- IPC: `/workspace/ipc/` 디렉토리 구조 동일

## 라이선스

MIT
