"""Demo script for HybridMemoryAgent (Bonus Challenge).

Demonstrates hybrid memory recall combining episodic memory (Qdrant) and
user profile features (Feast).

Runs 5 distinct queries showcasing:
  1. Direct episodic memory recall (vector hit)
  2. Profile-guided recommendation query (topic_affinity & reading speed)
  3. Real-time velocity query (queries_last_hour)
  4. Vietnamese paraphrased semantic search
  5. Mixed hybrid recall (episodic grounding + user profile context)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bonus.agent import HybridMemoryAgent


def main() -> None:
    print("=" * 70)
    print("DAY 19 BONUS CHALLENGE: HYBRID MEMORY AI ASSISTANT DEMO")
    print("=" * 70)

    # 1. Initialize HybridMemoryAgent
    print("\n[1/3] Initializing HybridMemoryAgent...")
    agent = HybridMemoryAgent(
        collection_name="bonus_episodic_memory",
        feast_repo_path="app/feast_repo",
    )
    print("  -> FastEmbed embedding loaded")
    print("  -> In-memory Qdrant Vector Store initialized")
    if agent.feature_store is not None:
        print("  -> Feast Feature Store connected (app/feast_repo)")
    else:
        print("  -> Feast Feature Store running with fallback schema")

    # 2. Seed Episodic Memories for user u_001
    print("\n[2/3] Seeding episodic memories for user 'u_001'...")
    memories = [
        (
            "Ghi chú Kubernetes: Cluster k8s sử dụng Horizontal Pod Autoscaler (HPA) để tự động co giãn "
            "số lượng pod dựa trên mức sử dụng CPU và custom metrics từ Prometheus.",
            {"topic": "cloud", "source": "k8s_docs", "importance": "high"},
        ),
        (
            "Ghi chú kiến trúc: Triển khai chiến lược Blue-Green Deployment và Canary Release "
            "trên AWS EKS giúp zero-downtime khi nâng cấp microservices.",
            {"topic": "cloud", "source": "aws_architecture", "importance": "medium"},
        ),
        (
            "Bảo mật Cloud: Bắt buộc kích hoạt AWS KMS để mã hóa dữ liệu tại chỗ (encryption at rest) "
            "trên S3 và RDS, tuân thủ nguyên tắc least-privilege với IAM roles và ghi log qua CloudTrail.",
            {"topic": "security", "source": "security_audit", "importance": "high"},
        ),
        (
            "Kiến trúc AI/ML: Hệ thống hybrid memory kết hợp Qdrant Vector Store (episodic memory) "
            "với Feast Feature Store (user profile) giúp tối ưu độ trễ P99 < 10ms và cá nhân hoá câu trả lời.",
            {"topic": "ai_ml", "source": "research_note", "importance": "high"},
        ),
        (
            "Ghi chú DevOps: Tối ưu hoá CI/CD pipeline với GitHub Actions và multi-stage Docker build "
            "giúp giảm dung lượng image từ 1.2GB xuống còn 180MB.",
            {"topic": "devops", "source": "devops_guide", "importance": "medium"},
        ),
    ]

    for text, meta in memories:
        mem_id = agent.remember(text=text, metadata=meta, user_id="u_001")
        print(f"  + Recorded memory [{mem_id}] (Topic: {meta['topic']})")

    # Also seed a memory for another user (u_002) to verify multi-tenant filtering
    agent.remember(
        text="Ghi chú Frontend: Tối ưu Largest Contentful Paint (LCP) với Next.js và CDN caching.",
        metadata={"topic": "frontend"},
        user_id="u_002",
    )

    # 3. Execute 5 Demonstration Queries
    print("\n[3/3] Executing 5 Demonstration Queries...")
    print("=" * 70)

    queries = [
        (
            1,
            "Hỏi đơn giản (Episodic Recall / Vector Hit)",
            "Tôi đã đọc gì về Kubernetes?",
            "u_001",
        ),
        (
            2,
            "Hỏi cần Profile Context (Topic Affinity & Reading Speed)",
            "Recommend đọc gì tiếp theo cho tôi?",
            "u_001",
        ),
        (
            3,
            "Hỏi cần Fresh Velocity Context (Queries Last Hour)",
            "Tôi đang quan tâm gì và hoạt động như thế nào gần đây?",
            "u_001",
        ),
        (
            4,
            "Hỏi Paraphrase tiếng Việt (Semantic Vector Wins)",
            "Tài liệu về tự động mở rộng hạ tầng khi tải cao?",
            "u_001",
        ),
        (
            5,
            "Hỏi Mixed (Episodic Memories + User Profile)",
            "Cho tôi summary cloud security và lưu ý áp dụng?",
            "u_001",
        ),
    ]

    for q_id, q_desc, query_str, u_id in queries:
        print(f"\n>>> QUERY #{q_id}: [{q_desc}]")
        print(f"User ID: '{u_id}' | Query: \"{query_str}\"")
        context = agent.recall(query=query_str, user_id=u_id, top_k=3)
        print(context)
        print("-" * 70)

    print("\n[SUCCESS] Demo completed successfully with 5 query outputs.")


if __name__ == "__main__":
    main()
