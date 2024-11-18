from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import GoldPrice,NewsArticle

# Register your models here.
class GoldPriceAdmin(admin.ModelAdmin):
    class Meta:
        model = GoldPrice
    list_display = ('date', 'price')
    list_filter = ('date',)
    search_fields = ('date',)

class NewsAdmin(admin.ModelAdmin):
    class Meta:
        model = NewsArticle
    list_display = ('published_at', 'title')
    list_filter = ('published_at',)
    search_fields = ('published_at',)


admin.site.register(GoldPrice, GoldPriceAdmin)
admin.site.register(NewsArticle, NewsAdmin)