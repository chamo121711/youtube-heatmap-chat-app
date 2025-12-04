# app.py (独立型ツール群)
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import base64
import os
import traceback
import subprocess
import json
from werkzeug.utils import secure_filename
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["TEMP_FOLDER"] = "temp_files"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["TEMP_FOLDER"], exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ========================================
# ツール1: チャットダウンローダー（yt-dlp使用）
# ========================================


@app.route("/download_chat", methods=["POST"])
def download_chat():
    """yt-dlpを使ってチャットを取得してExcelファイルとして返す"""
    try:
        data = request.get_json()
        video_url = data.get("video_url", "")

        if not video_url:
            return jsonify({"success": False, "message": "動画URLを入力してください"})

        print(f"=== チャットダウンロード開始（yt-dlp）: {video_url} ===")

        # 動画IDを抽出
        video_id_match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", video_url)
        if not video_id_match:
            return jsonify(
                {"success": False, "message": "有効なYouTube URLではありません"}
            )

        video_id = video_id_match.group(1)

        # 一時ファイル名
        temp_base = os.path.join(app.config["TEMP_FOLDER"], f"chat_{video_id}")
        chat_json_file = f"{temp_base}.live_chat.json"

        # 既存のファイルを削除
        if os.path.exists(chat_json_file):
            os.remove(chat_json_file)

        print("yt-dlpでチャット取得中...")

        # yt-dlpでチャットをダウンロード
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--write-subs",
                    "--sub-lang",
                    "live_chat",
                    "--skip-download",
                    "--output",
                    temp_base,
                    video_url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                error_msg = result.stderr
                print(f"yt-dlpエラー: {error_msg}")

                # エラーメッセージを解析
                if (
                    "Requested format is not available" in error_msg
                    or "live chat" in error_msg.lower()
                ):
                    return jsonify(
                        {
                            "success": False,
                            "message": "この動画にはチャットデータがありません。\n\nライブ配信ではない、またはチャットが無効になっている可能性があります。",
                        }
                    )
                else:
                    return jsonify(
                        {
                            "success": False,
                            "message": f"チャット取得エラー:\n{error_msg[:200]}",
                        }
                    )

        except subprocess.TimeoutExpired:
            return jsonify(
                {
                    "success": False,
                    "message": "タイムアウト: チャットの取得に時間がかかりすぎています。動画が長すぎるか、チャットが多すぎる可能性があります。",
                }
            )
        except FileNotFoundError:
            return jsonify(
                {
                    "success": False,
                    "message": "yt-dlpがインストールされていません。\n\nコマンドラインで以下を実行してください:\npip install yt-dlp",
                }
            )

        # JSONファイルが生成されたか確認
        if not os.path.exists(chat_json_file):
            return jsonify(
                {
                    "success": False,
                    "message": "チャットデータが取得できませんでした。この動画にはチャットがない可能性があります。",
                }
            )

        print(f"JSONファイル取得成功: {chat_json_file}")

        # JSONファイルをパースしてDataFrameに変換
        chat_data = []

        try:
            with open(chat_json_file, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    try:
                        line_count += 1
                        if line_count % 1000 == 0:
                            print(f"処理中... {line_count}行")

                        json_data = json.loads(line)

                        # replayChatItemAction形式のデータを処理
                        if "replayChatItemAction" in json_data:
                            actions = json_data["replayChatItemAction"].get(
                                "actions", []
                            )

                            for action in actions:
                                if "addChatItemAction" in action:
                                    item = action["addChatItemAction"].get("item", {})

                                    # 通常のチャットメッセージ
                                    if "liveChatTextMessageRenderer" in item:
                                        msg = item["liveChatTextMessageRenderer"]

                                        # タイムスタンプ（マイクロ秒）
                                        time_usec = int(msg.get("timestampUsec", 0))
                                        time_seconds = time_usec / 1000000

                                        hours = int(time_seconds // 3600)
                                        minutes = int((time_seconds % 3600) // 60)
                                        seconds = int(time_seconds % 60)

                                        # 作者名
                                        author = ""
                                        if "authorName" in msg:
                                            if "simpleText" in msg["authorName"]:
                                                author = msg["authorName"]["simpleText"]

                                        # メッセージ
                                        message_text = ""
                                        if "message" in msg:
                                            if "runs" in msg["message"]:
                                                message_text = "".join(
                                                    [
                                                        run.get("text", "")
                                                        for run in msg["message"][
                                                            "runs"
                                                        ]
                                                    ]
                                                )
                                            elif "simpleText" in msg["message"]:
                                                message_text = msg["message"][
                                                    "simpleText"
                                                ]

                                        chat_data.append(
                                            {
                                                "date": datetime.now().strftime(
                                                    "%Y-%m-%d"
                                                ),
                                                "hour": hours,
                                                "minute": minutes,
                                                "second": seconds,
                                                "author": author,
                                                "message": message_text,
                                                "amountString": "",
                                            }
                                        )

                                    # スーパーチャット
                                    elif "liveChatPaidMessageRenderer" in item:
                                        msg = item["liveChatPaidMessageRenderer"]

                                        time_usec = int(msg.get("timestampUsec", 0))
                                        time_seconds = time_usec / 1000000

                                        hours = int(time_seconds // 3600)
                                        minutes = int((time_seconds % 3600) // 60)
                                        seconds = int(time_seconds % 60)

                                        author = ""
                                        if "authorName" in msg:
                                            if "simpleText" in msg["authorName"]:
                                                author = msg["authorName"]["simpleText"]

                                        message_text = ""
                                        if "message" in msg:
                                            if "runs" in msg["message"]:
                                                message_text = "".join(
                                                    [
                                                        run.get("text", "")
                                                        for run in msg["message"][
                                                            "runs"
                                                        ]
                                                    ]
                                                )

                                        amount = msg.get("purchaseAmountText", {}).get(
                                            "simpleText", ""
                                        )

                                        chat_data.append(
                                            {
                                                "date": datetime.now().strftime(
                                                    "%Y-%m-%d"
                                                ),
                                                "hour": hours,
                                                "minute": minutes,
                                                "second": seconds,
                                                "author": author,
                                                "message": message_text,
                                                "amountString": amount,
                                            }
                                        )

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue

        except Exception as e:
            print(f"JSONパースエラー: {e}")
            return jsonify(
                {
                    "success": False,
                    "message": f"チャットデータの解析に失敗しました: {str(e)}",
                }
            )

        finally:
            # 一時ファイルを削除
            try:
                if os.path.exists(chat_json_file):
                    os.remove(chat_json_file)
            except:
                pass

        if not chat_data:
            return jsonify(
                {"success": False, "message": "チャットデータが取得できませんでした。"}
            )

        print(f"✓ {len(chat_data)}件のチャットを取得しました")

        # DataFrameに変換してExcelファイルを作成
        df = pd.DataFrame(chat_data)

        # メモリ上でExcelファイルを作成
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Chat")
        output.seek(0)

        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=f"chat_{video_id}.xlsx",
        )

    except Exception as e:
        print(f"予期しないエラー: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"エラー: {str(e)}"})


# ========================================
# ツール2: ヒートマップダウンローダー
# ========================================


@app.route("/download_heatmap", methods=["POST"])
def download_heatmap():
    """ヒートマップのみを取得してテキストファイルとして返す"""
    try:
        data = request.get_json()
        video_url = data.get("video_url", "")

        if not video_url:
            return jsonify({"success": False, "message": "動画URLを入力してください"})

        print(f"=== ヒートマップダウンロード開始: {video_url} ===")

        # 動画IDを抽出
        video_id_match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", video_url)
        if not video_id_match:
            return jsonify(
                {"success": False, "message": "有効なYouTube URLではありません"}
            )

        video_id = video_id_match.group(1)

        # Seleniumでヒートマップを取得
        try:
            print("ヒートマップ取得中...")

            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")

            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )

            driver.get(video_url)
            time.sleep(8)  # ページ読み込み待機

            page_source = driver.page_source
            driver.quit()

            # ヒートマップのpathを抽出
            match = re.search(r'class="ytp-heat-map-path"[^>]*d="([^"]+)"', page_source)
            if not match:
                match = re.search(
                    r'class="ytp-modern-heat-map"[^>]*d="([^"]+)"', page_source
                )

            if not match:
                return jsonify(
                    {
                        "success": False,
                        "message": "ヒートマップデータが見つかりませんでした。\n\nこの動画にはヒートマップがない可能性があります。",
                    }
                )

            path_data = match.group(1)

            # テキストファイルとして返す
            full_path_tag = f'<path class="ytp-modern-heat-map" d="{path_data}"></path>'

            output = io.BytesIO(full_path_tag.encode("utf-8"))
            output.seek(0)

            print(f"✓ ヒートマップデータを取得しました")

            return send_file(
                output,
                mimetype="text/plain",
                as_attachment=True,
                download_name=f"heatmap_{video_id}.txt",
            )

        except Exception as e:
            error_msg = str(e)
            print(f"エラー: {error_msg}")
            return jsonify(
                {"success": False, "message": f"ヒートマップ取得エラー: {error_msg}"}
            )

    except Exception as e:
        print(f"予期しないエラー: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"エラー: {str(e)}"})


# ========================================
# ツール3: データ分析ツール
# ========================================


class YouTubeChatHeatmapAnalyzer:
    def __init__(self):
        self.chat_df = None
        self.heatmap_data = None
        self.merged_data = None
        self.heatmap_max_x = 0
        self.surge_analysis = None

    def load_chat_data(self, file_path):
        try:
            self.chat_df = pd.read_excel(file_path, engine="openpyxl")
            return True, f"{len(self.chat_df)}行のチャットデータを読み込みました"
        except Exception as e:
            return False, f"エラー: {str(e)}"

    def load_heatmap_data(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            match = re.search(r'd="([^"]+)"', content)
            if match:
                path_data = match.group(1)
                self.heatmap_data = self._parse_heatmap_path(path_data)

                if self.heatmap_data is not None and len(self.heatmap_data) > 0:
                    return (
                        True,
                        f"{len(self.heatmap_data)}個のヒートマップポイントを読み込みました",
                    )
                else:
                    return False, "ヒートマップデータのパースに失敗しました"
            else:
                return False, 'd="..." 形式のpathデータが見つかりませんでした'

        except Exception as e:
            return False, f"エラー: {str(e)}"

    def _parse_heatmap_path(self, path_string):
        try:
            segments = path_string.split(" C ")
            points = []

            for segment in segments:
                numbers = re.findall(r"[\d.]+", segment)
                if len(numbers) >= 6:
                    x = float(numbers[-2])
                    y = float(numbers[-1])
                    points.append({"x": x, "y": 100 - y})

            if not points:
                return None

            max_x = max(p["x"] for p in points)
            self.heatmap_max_x = max_x

            time_series = []
            for p in points:
                time_series.append({"time": p["x"], "heat_intensity": p["y"]})

            return pd.DataFrame(time_series)

        except Exception as e:
            print(f"パースエラー: {e}")
            return None

    def process_chat_data(self):
        if self.chat_df is None:
            return None

        try:
            self.chat_df["total_seconds"] = (
                self.chat_df["hour"].fillna(0).astype(int) * 3600
                + self.chat_df["minute"].fillna(0).astype(int) * 60
                + self.chat_df["second"].fillna(0).astype(int)
            )

            start_time = self.chat_df["total_seconds"].min()
            self.chat_df["elapsed_seconds"] = self.chat_df["total_seconds"] - start_time

            self.chat_df["time_bucket"] = (self.chat_df["elapsed_seconds"] // 10) * 10

            chat_counts = self.chat_df.groupby("time_bucket").size().reset_index()
            chat_counts.columns = ["time", "chat_count"]

            return chat_counts

        except Exception as e:
            print(f"チャットデータ処理エラー: {e}")
            return None

    def calculate_surge_rate(self):
        if self.merged_data is None:
            return None

        try:
            df = self.merged_data.copy()

            df["moving_avg"] = df["chat_count"].rolling(window=3, min_periods=1).mean()
            df["surge_vs_moving_avg"] = (
                (df["chat_count"] - df["moving_avg"]) / (df["moving_avg"] + 1)
            ) * 100

            df["prev_count"] = df["chat_count"].shift(1).fillna(0)
            df["surge_vs_prev"] = (
                (df["chat_count"] - df["prev_count"]) / (df["prev_count"] + 1)
            ) * 100

            overall_avg = df["chat_count"].mean()
            df["surge_vs_overall"] = (
                (df["chat_count"] - overall_avg) / (overall_avg + 1)
            ) * 100

            df["surge_rate_combined"] = (
                df["surge_vs_moving_avg"] + df["surge_vs_prev"] + df["surge_vs_overall"]
            ) / 3

            self.surge_analysis = df

            top_surges = df.nlargest(5, "surge_rate_combined")[
                ["time", "time_label", "chat_count", "surge_rate_combined"]
            ]

            return top_surges

        except Exception as e:
            print(f"上昇率計算エラー: {e}")
            return None

    def merge_data(self):
        if self.chat_df is None or self.heatmap_data is None:
            return False, "チャットデータとヒートマップデータの両方が必要です"

        try:
            chat_counts = self.process_chat_data()

            if chat_counts is None:
                return False, "チャットデータの処理に失敗しました"

            actual_video_length = self.chat_df["elapsed_seconds"].max()

            self.heatmap_data["time_adjusted"] = (
                self.heatmap_data["time"] / self.heatmap_max_x
            ) * actual_video_length

            self.heatmap_data["time_bucket"] = (
                self.heatmap_data["time_adjusted"] // 10
            ) * 10
            heatmap_avg = (
                self.heatmap_data.groupby("time_bucket")["heat_intensity"]
                .mean()
                .reset_index()
            )
            heatmap_avg.columns = ["time", "heat_intensity"]

            max_time = max(
                chat_counts["time"].max() if len(chat_counts) > 0 else 0,
                heatmap_avg["time"].max() if len(heatmap_avg) > 0 else 0,
            )

            all_times = pd.DataFrame({"time": range(0, int(max_time) + 10, 10)})

            self.merged_data = all_times.merge(chat_counts, on="time", how="left")
            self.merged_data = self.merged_data.merge(
                heatmap_avg, on="time", how="left"
            )

            self.merged_data["chat_count"] = self.merged_data["chat_count"].fillna(0)
            self.merged_data["heat_intensity"] = self.merged_data[
                "heat_intensity"
            ].fillna(0)

            self.merged_data["time_label"] = self.merged_data["time"].apply(
                lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
            )

            self.calculate_surge_rate()

            return True, f"{len(self.merged_data)}個のデータポイントを統合しました"

        except Exception as e:
            return False, f"データ統合エラー: {str(e)}"

    def generate_graph(self):
        if self.merged_data is None or self.surge_analysis is None:
            return None

        try:
            fig, axes = plt.subplots(4, 1, figsize=(16, 14))
            fig.suptitle(
                "YouTube Chat & Heatmap Analysis",
                fontsize=20,
                fontweight="bold",
                y=0.995,
            )

            top_surges = self.surge_analysis.nlargest(5, "surge_rate_combined")
            top_times = top_surges["time"].values

            # グラフ1: 統合グラフ（マーカー付き）
            ax1 = axes[0]
            ax1_twin = ax1.twinx()

            ax1.bar(
                self.merged_data["time"],
                self.merged_data["chat_count"],
                alpha=0.7,
                color="#3b82f6",
                label="Chat Count",
                width=8,
            )

            for t in top_times:
                surge_row = self.surge_analysis[self.surge_analysis["time"] == t]
                if not surge_row.empty:
                    ax1.scatter(
                        t,
                        surge_row["chat_count"].values[0],
                        color="red",
                        s=200,
                        marker="*",
                        zorder=5,
                        edgecolors="yellow",
                        linewidth=2,
                    )

            ax1.set_ylabel(
                "Chat Count", fontsize=12, color="#3b82f6", fontweight="bold"
            )
            ax1.tick_params(axis="y", labelcolor="#3b82f6")
            ax1.grid(True, alpha=0.3, linestyle="--")

            ax1_twin.fill_between(
                self.merged_data["time"],
                self.merged_data["heat_intensity"],
                alpha=0.3,
                color="#ef4444",
                label="Heat Intensity",
            )
            ax1_twin.plot(
                self.merged_data["time"],
                self.merged_data["heat_intensity"],
                color="#ef4444",
                linewidth=1,
            )
            ax1_twin.set_ylabel(
                "Heat Intensity", fontsize=12, color="#ef4444", fontweight="bold"
            )
            ax1_twin.tick_params(axis="y", labelcolor="#ef4444")

            ax1.set_title(
                "Integrated: Chat Count & Heat Intensity (★=Top Surge)",
                fontsize=14,
                fontweight="bold",
                pad=10,
            )
            ax1.set_xlabel("Time (seconds)", fontsize=11)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9
            )

            # グラフ2: チャット数のみ
            ax2 = axes[1]
            ax2.bar(
                self.merged_data["time"],
                self.merged_data["chat_count"],
                color="#3b82f6",
                alpha=0.8,
                width=8,
            )
            ax2.set_title(
                "Chat Count Over Time", fontsize=14, fontweight="bold", pad=10
            )
            ax2.set_ylabel("Chat Count", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Time (seconds)", fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle="--")

            # グラフ3: ヒートマップのみ
            ax3 = axes[2]
            ax3.fill_between(
                self.merged_data["time"],
                0,
                self.merged_data["heat_intensity"],
                alpha=0.4,
                color="#ef4444",
            )
            ax3.plot(
                self.merged_data["time"],
                self.merged_data["heat_intensity"],
                color="#ef4444",
                linewidth=1,
            )
            ax3.set_title(
                "Heat Intensity Over Time", fontsize=14, fontweight="bold", pad=10
            )
            ax3.set_ylabel("Heat Intensity", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Time (seconds)", fontsize=11)
            ax3.grid(True, alpha=0.3, linestyle="--")
            ax3.set_ylim(bottom=0)

            # グラフ4: 上昇率
            ax4 = axes[3]
            ax4.plot(
                self.surge_analysis["time"],
                self.surge_analysis["surge_vs_prev"],
                label="vs Previous (10s ago)",
                color="#10b981",
                linewidth=2,
                alpha=0.7,
            )
            ax4.plot(
                self.surge_analysis["time"],
                self.surge_analysis["surge_vs_moving_avg"],
                label="vs Moving Avg (30s)",
                color="#f59e0b",
                linewidth=2,
                alpha=0.7,
            )
            ax4.plot(
                self.surge_analysis["time"],
                self.surge_analysis["surge_vs_overall"],
                label="vs Overall Avg",
                color="#8b5cf6",
                linewidth=2,
                alpha=0.7,
            )
            ax4.axhline(y=0, color="gray", linestyle="--", linewidth=1)

            for t in top_times:
                ax4.axvline(x=t, color="red", linestyle=":", alpha=0.5, linewidth=1)

            ax4.set_title(
                "Chat Surge Rate Analysis", fontsize=14, fontweight="bold", pad=10
            )
            ax4.set_ylabel("Surge Rate (%)", fontsize=11, fontweight="bold")
            ax4.set_xlabel("Time (seconds)", fontsize=11)
            ax4.legend(loc="upper right", framealpha=0.9)
            ax4.grid(True, alpha=0.3, linestyle="--")

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            print(f"グラフ生成エラー: {e}")
            return None

    def get_statistics(self):
        if self.merged_data is None or self.surge_analysis is None:
            return None

        correlation = self.merged_data["chat_count"].corr(
            self.merged_data["heat_intensity"]
        )

        top_surges = self.surge_analysis.nlargest(5, "surge_rate_combined")
        top_surges_list = []
        for _, row in top_surges.iterrows():
            top_surges_list.append(
                {
                    "time": int(row["time"]),
                    "time_label": row["time_label"],
                    "chat_count": int(row["chat_count"]),
                    "surge_rate": float(row["surge_rate_combined"]),
                }
            )

        stats = {
            "total_chats": int(self.merged_data["chat_count"].sum()),
            "max_chat_count": int(self.merged_data["chat_count"].max()),
            "avg_chat_count": float(self.merged_data["chat_count"].mean()),
            "avg_heat_intensity": float(self.merged_data["heat_intensity"].mean()),
            "max_heat_intensity": float(self.merged_data["heat_intensity"].max()),
            "data_points": len(self.merged_data),
            "correlation": float(correlation),
            "top_surges": top_surges_list,
        }

        return stats


analyzer = YouTubeChatHeatmapAnalyzer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """アップロードされたファイルを分析"""
    global analyzer
    analyzer = YouTubeChatHeatmapAnalyzer()

    try:
        print("=== データ分析開始 ===")

        if "chat_file" not in request.files:
            return jsonify(
                {"success": False, "message": "チャットファイルが選択されていません"}
            )

        chat_file = request.files["chat_file"]
        if chat_file.filename == "":
            return jsonify(
                {"success": False, "message": "チャットファイルが選択されていません"}
            )

        if "heatmap_file" not in request.files:
            return jsonify(
                {
                    "success": False,
                    "message": "ヒートマップファイルが選択されていません",
                }
            )

        heatmap_file = request.files["heatmap_file"]
        if heatmap_file.filename == "":
            return jsonify(
                {
                    "success": False,
                    "message": "ヒートマップファイルが選択されていません",
                }
            )

        chat_filename = secure_filename(chat_file.filename)
        heatmap_filename = secure_filename(heatmap_file.filename)

        chat_path = os.path.join(app.config["UPLOAD_FOLDER"], chat_filename)
        heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)

        chat_file.save(chat_path)
        heatmap_file.save(heatmap_path)

        success, message = analyzer.load_chat_data(chat_path)
        if not success:
            return jsonify(
                {
                    "success": False,
                    "message": f"チャットデータ読み込みエラー: {message}",
                }
            )

        success, message = analyzer.load_heatmap_data(heatmap_path)
        if not success:
            return jsonify(
                {
                    "success": False,
                    "message": f"ヒートマップデータ読み込みエラー: {message}",
                }
            )

        success, message = analyzer.merge_data()
        if not success:
            return jsonify(
                {"success": False, "message": f"データ統合エラー: {message}"}
            )

        graph_base64 = analyzer.generate_graph()
        if graph_base64 is None:
            return jsonify({"success": False, "message": "グラフの生成に失敗しました"})

        stats = analyzer.get_statistics()

        try:
            os.remove(chat_path)
            os.remove(heatmap_path)
        except:
            pass

        print("=== 分析完了 ===")

        return jsonify(
            {
                "success": True,
                "message": "データの分析が完了しました",
                "graph": graph_base64,
                "stats": stats,
            }
        )

    except Exception as e:
        print(f"分析エラー: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"エラー: {str(e)}"})


@app.route("/download_csv")
def download_csv():
    """統合データをCSVでダウンロード"""
    global analyzer

    if analyzer.merged_data is None:
        return "データがありません", 400

    try:
        output = io.StringIO()
        analyzer.merged_data.to_csv(output, index=False, encoding="utf-8-sig")
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="youtube_analysis.csv",
        )
    except Exception as e:
        return f"エラー: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
