from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render, redirect
from django.db.models import Q
from .models import Anime, Myrating, MyList
from .forms import UserRegisterForm
from django.contrib import messages
# Create your views here.


def info(request):
    return render(request, 'anime_recommend/home.html')

def index(request):
    # fetch all the animes stored in the database
    animes = Anime.objects.all()
    query = request.GET.get('q')

    if request.user.is_authenticated:
        messages.success(
            request, f'Now click profile to see your watch list')
    else:
        messages.success(
            request, f'Register to enjopy our anime recommendation service!')

    if query:
        animes = Anime.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'anime_recommend/list.html', {'animes': animes})

    return render(request, 'anime_recommend/list.html', {'animes': animes})

# Register user
def register(request):
    # if user have clicked the submitted for signning up 
    if request.method == "POST":

        form = UserRegisterForm(request.POST)
        # check if the form is valid
        if form.is_valid():
            
            # save this user created from this form
            form.save()
            # get user name
            username = form.cleaned_data['username']
            # display a flash message
            messages.success(request, f'Account created for {username}! You are now able to login')
            return redirect("login")
    else:
        form = UserRegisterForm()
    

    context = {'form': form}

    return render(request, 'anime_recommend/register.html', context)

@login_required
def profile(request):
    animes = Anime.objects.filter(mylist__watch=True, mylist__user=request.user)
    query = request.GET.get('q')

    if query:
        animes = Anime.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'anime_recommend/profile.html', {'animes': animes})

    return render(request, 'anime_recommend/profile.html', {'animes': animes})

