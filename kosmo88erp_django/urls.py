from django.contrib import admin
from django.conf.urls import url
from django.urls import path, include

from django.views.generic import TemplateView

from django.conf import settings
from django.conf.urls.static import static

# Loading plotly Dash apps script
import kosmo88erp_django.dash_app_code

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^django_plotly_dash/', include('django_plotly_dash.urls')),
    path('', TemplateView.as_view(template_name='dash_plot.html'), name='home'),
]
