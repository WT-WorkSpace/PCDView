from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout


HELP_MANUAL_HTML = """
<h2>Point Cloud Viewer 功能说明</h2>

<h3>文件与数据</h3>
<ul>
  <li><b>Open File</b>：打开单个 PCD 点云文件。</li>
  <li><b>Open Directory</b>：打开点云目录，支持按帧播放和切换。</li>
  <li><b>Open BBoxes Dir</b>：选择目标框 JSON 标注目录。</li>
  <li><b>导入历史帧</b>：导入历史点云目录，导入成功后按钮蓝底表示已启用。</li>
</ul>

<h3>播放与切帧</h3>
<ul>
  <li>底部播放栏用于上一帧、播放/暂停、下一帧和拖动帧进度。</li>
  <li><b>Z</b>：上一帧；长按 Z 连续向前播放。</li>
  <li><b>X</b>：下一帧；长按 X 连续向后播放。</li>
  <li>底部帧信息固定显示 35 个字符，完整文件名可悬停查看。</li>
</ul>

<h3>视图与显示</h3>
<ul>
  <li><b>Point Size + / -</b>：增大或减小点云点大小，最小点大小为 1。</li>
  <li><b>Color</b>：选择点云颜色或字段映射颜色。</li>
  <li><b>Coordinate</b>：显示或隐藏坐标轴。</li>
  <li><b>Save View / Load View</b>：保存和加载当前视角。</li>
</ul>

<h3>标注与编辑</h3>
<ul>
  <li><b>标注3D框</b>：在主视图拖拽矩形区域生成 3D 目标框。</li>
  <li>点击目标框会打开右侧三视图和目标框属性面板。</li>
  <li><b>Backspace</b>：删除当前选中的目标框。</li>
  <li><b>C</b>：将当前选中目标框 yaw 旋转 90 度。</li>
  <li><b>Space</b>：切换到下一个目标框三视图。</li>
  <li><b>Save</b>：保存当前帧标注；<b>Copy Prev</b>：复制上一帧标注。</li>
  <li><b>自定义标注属性</b>：配置目标框属性字段和历史帧显示模式。</li>
</ul>

<h3>点云框选</h3>
<ul>
  <li><b>点云框选</b>：开启后在主视图拖拽一次矩形，选中点会高亮。</li>
  <li>框选完成后左侧显示点字段表格，不会挤占底部播放条。</li>
  <li><b>取消框选</b>：清除高亮并隐藏框选结果表格。</li>
</ul>

<h3>辅助功能</h3>
<ul>
  <li><b>添加/取消平面</b>：显示或移除参考平面；启用时按钮蓝底。</li>
  <li><b>显示/关闭 Mask</b>：显示 Mask 点线或按 Mask 过滤点云。</li>
  <li><b>点云聚类</b>：对当前帧执行聚类并绘制聚类框；启用时按钮蓝底。</li>
  <li><b>外参标定</b>：打开外参标定面板，对多雷达点云进行位姿调整和应用。</li>
</ul>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Cloud Viewer 功能说明")
        self.resize(760, 620)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setHtml(HELP_MANUAL_HTML)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(browser, 1)
        layout.addWidget(buttons)
