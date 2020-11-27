################################
#機械学習アプリを簡単に作成できるモジュールGradio
# https://github.com/gradio-app/gradio
# https://cpp-learning.com/gradio/#Step1Google_Colaboratory
# https://www.gradio.app
# 操作感としてはTKinterに近いものがある。Tkinterより自由度は低いものの、UIは結構かっこいい。
################################

import gradio as gr

def greet(name):
  return "Hello " + name + "!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch(share=False)

#これはWebCameraも動かせるそうなので、動画の読み込み＋処理を走らせるというアプリケーションとして
#有能な可能性大！！！


