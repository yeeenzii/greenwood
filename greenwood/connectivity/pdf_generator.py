import jinja2
import pdfkit

elephant_f = 83
motorbike_f = 93
gunshot_f = 50
human_f = 0
logging_f = 0
logging_o = 0
poaching_o = 0


context = {'ef': elephant_f, 'mf': motorbike_f, 
           'gf': gunshot_f, 'hf': human_f, 'lf' : logging_f, 
           'lo' : logging_o, 'po' : poaching_o}

template_loader = jinja2.FileSystemLoader('./')
template_env = jinja2.Environment(loader=template_loader)

file_template = r'pdf-template.html'
template = template_env.get_template(file_template)
output_text = template.render(context)

config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
pdfkit.from_string(output_text, 'generated_data.pdf', configuration=config, css = 'style.css')