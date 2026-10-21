from django import template
register = template.Library()

@register.filter
def get_item(d, key):
    return d.get(key, {})

@register.filter
def index(seq, i):
    try:
        return seq[i]
    except (IndexError, TypeError):
        return ''