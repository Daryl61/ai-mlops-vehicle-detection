import sys
sys.path.insert(0, ".")
from stream import stream_video
from detect import detect_objects

from api.app import app
from fastapi.testclient import TestClient
import yaml
config  = yaml.safe_load(open("configs/config.yaml", "r"))
import pytest

    
client = TestClient(app)

def test_api_health():

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_api_detect():
    response = client.post("/detect", files={"file": open("tests/test_image.jpg", "rb")})
    assert response.status_code == 200

def test_api_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    assert "raw_count" in response.json()
    assert "labeled_count" in response.json()

@pytest.mark.skip(reason="CI'da kamera yok")
def test_api_detect_live():
    response = client.get("/detect-live")
    assert response.status_code == 200
    assert "detections" in response.json()


def test_stream_video():
    frames = []
    for i, frame in enumerate(stream_video()):
        frames.append(frame)
        if i >= 4:  
            break
    assert len(frames) == 5

def test_detect_objects():
    frame = next(stream_video())
    detections = detect_objects(frame)
    assert isinstance(detections, list)    