import sys
from OpenGL import GLUT
from OpenGL.error import GLError

import os

from OpenGL import platform

def display():
    pass

def log():
    print("OpenGL context is active and functioning.")

try:
    print("Initializing GLUT...")
    print(os.environ["PYOPENGL_PLATFORM"])
    
    GLUT.glutInit(sys.argv)
    GLUT.glutCreateWindow(b"Minimal OpenGL Test")
    print(platform.GetCurrentContext())
    print("✅ Success! Window and context were created.")
    print("Registering timer function...")
    GLUT.glutTimerFunc(100, lambda x: log(), 0)
    # print("Activating OpenGL context...")
    GLUT.glutMainLoop()

except GLError as e:
    print(f"❌ Error: An OpenGL error occurred. This is likely an environment issue.")
    print(f"Details: {e}")
except Exception as e:
    print(f"❌ Error: A general error occurred during initialization.")
    print(f"Details: {e}")
    raise e
    