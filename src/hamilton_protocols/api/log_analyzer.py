from datetime import datetime
from collections import defaultdict
import json
import os
from typing import Dict, List


def parse_timestamp(line: str) -> datetime:
    """Parse timestamp from log line."""
    timestamp_str = line.split(">")[0].strip()
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


def extract_command_type(line: str) -> str:
    """Extract command type from log line."""
    if "Response Content:" in line:
        try:
            content = line.split("Response Content:")[1].strip()
            if content.startswith("{"):
                data = json.loads(content)
                return data.get("command", "unknown")
        except Exception as e:
            print(f"Error parsing command type: {str(e)}")
    return None


def analyze_log_file(file_path: str) -> Dict[str, List[float]]:
    """Analyze log file and return command execution times."""
    command_times = defaultdict(list)
    current_command = None
    start_time = None

    # Try different encodings
    encodings = ["utf-8", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line in f:
                    # Look for GET request that starts a command
                    if "HttpGET - progress" in line and "Response Content:" in line:
                        current_command = extract_command_type(line)
                        if current_command:
                            start_time = parse_timestamp(line)

                    # Look for POST request that ends a command
                    elif (
                        "HttpPOST - progress" in line and current_command and start_time
                    ):
                        end_time = parse_timestamp(line)
                        duration = (end_time - start_time).total_seconds()
                        command_times[current_command].append(duration)
                        current_command = None
                        start_time = None
            # If we get here, the file was read successfully
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {str(e)}")
            continue

    return command_times


def get_analysis(command_times: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Get analysis of command execution times as a dictionary."""
    analysis = {}

    for command, times in command_times.items():
        if times:
            analysis[command] = {
                "count": len(times),
                "average_time": round(sum(times) / len(times), 3),
                "min_time": round(min(times), 3),
                "max_time": round(max(times), 3),
                "total_time": round(sum(times), 3),
            }

    return analysis


def analyze_all_logs(log_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze all .trc files in a directory and return combined analysis."""
    all_analyses = {}

    # Get all .trc files in the directory
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".trc")]

    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        try:
            command_times = analyze_log_file(file_path)
            if command_times:
                analysis = get_analysis(command_times)
                all_analyses[log_file] = analysis
        except Exception as e:
            print(f"Error analyzing {log_file}: {str(e)}")

    return all_analyses


def get_combined_analysis(
    all_analyses: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Combine analyses from multiple log files into a single analysis."""
    combined = defaultdict(lambda: {"count": 0, "total_time": 0, "times": []})

    for file_analysis in all_analyses.values():
        for command, stats in file_analysis.items():
            combined[command]["count"] += stats["count"]
            combined[command]["total_time"] += stats["total_time"]
            # Store individual times for min/max calculation
            combined[command]["times"].extend([stats["min_time"], stats["max_time"]])

    # Calculate final statistics
    final_analysis = {}
    for command, stats in combined.items():
        final_analysis[command] = {
            "count": stats["count"],
            "average_time": round(stats["total_time"] / stats["count"], 3),
            "min_time": round(min(stats["times"]), 3),
            "max_time": round(max(stats["times"]), 3),
            "total_time": round(stats["total_time"], 3),
        }

    return final_analysis
