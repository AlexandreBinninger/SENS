from custom_types import *
import constants
from tqdm import tqdm
from utils import files_utils
import os
import options
import subprocess
import lxml.etree as ET
from PIL import Image as PILImage
import io
import cairosvg

SVG_tag_prefix ="{http://www.w3.org/2000/svg}"

class SVG:

    def render(self, width: int, height:int, output_filename:Optional[str] = None, background_color = "white"):
        bstr_xml = ET.tostring(self.root)
        png_data = cairosvg.svg2png(bytestring = bstr_xml, background_color = background_color, output_width= width, output_height=height)
        if output_filename is not None:
            with open(output_filename, "wb") as binary_file:
                binary_file.write(png_data)
        return png_data

    def change_width(self, new_weights:List[float]):
        if (len(new_weights) != self.num_strokes()):
            print(f"The list of weights has {len(new_weights)} elements and does not have the same size as the number of strokes ({self.num_strokes()} elements).")
        i=0
        for child in self.root:
            if (child.tag == f"{SVG_tag_prefix}g"):
                for grandchild in child:
                    if (grandchild.tag == f"{SVG_tag_prefix}path"):
                        attributes = grandchild.attrib
                        attributes["stroke-width"] = f"{new_weights[i%len(new_weights)]}"
                        i+=1

    def change_width_uniform(self, new_weight:float):
        for child in self.root:
            if (child.tag == f"{SVG_tag_prefix}g"):
                for grandchild in child:
                    if (grandchild.tag == f"{SVG_tag_prefix}path"):
                        attributes = grandchild.attrib
                        attributes["stroke-width"] = f"{new_weight}"
    
    def num_strokes(self):
        count = 0
        for child in self.root:
            if (child.tag == f"{SVG_tag_prefix}g"):
                for grandchild in child:
                    if (grandchild.tag == f"{SVG_tag_prefix}path"):
                        count+=1
        return count
    
    def tostring(self):
        return ET.tostring(self.root)

    def __init__(self, filename:str) -> None:
        self.filename = filename
        self.xml_tree = ET.parse(self.filename)
        self.root = self.xml_tree.getroot()
