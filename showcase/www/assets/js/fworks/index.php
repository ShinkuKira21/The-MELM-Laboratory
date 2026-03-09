<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Unauthorised Access (E:403)</title>
    <link rel="stylesheet" href="/assets/css/tailwind-styles.css">
    <link rel="icon" type="image/png" sizes="32x32" href="/assets/img/favicon_io/favicon-32x32.png">
    <link rel="icon" type="image/png" sizes="16x16" href="/assets/img/favicon_io/favicon-16x16.png">
    <meta name="robots" content="noindex, nofollow" />
</head>

<body class="bg-gray-900 text-gray-100 flex items-center justify-center min-h-screen p-6">
    <div class="max-w-lg w-full text-center">
        
        <div class="flex justify-center mb-6">
            <div class="bg-red-500/10 p-6 rounded-full border border-red-500/30">
                <svg xmlns="http://www.w3.org/2000/svg" 
                     class="h-16 w-16 text-red-400"
                     viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path stroke-linecap="round" stroke-linejoin="round" 
                        d="M12 9v3m0 3h.01M9.75 3h4.5l7.5 7.5-7.5 7.5h-4.5L2.25 10.5 9.75 3z" />
                </svg>
            </div>
        </div>

        <h1 class="text-4xl font-extrabold tracking-tight text-red-400 mb-4">
            Unauthorised Access
        </h1>

        <p class="text-gray-300 text-lg mb-8">
            You do not have permission to access this resource.<br/>
            If you believe this is an error, please contact the site administrator.
        </p>

        <a href="/" 
           class="inline-block px-6 py-3 rounded-md bg-red-600 hover:bg-red-700 
                  text-white font-medium transition">
            Return to Home
        </a>

        <div class="mt-8 text-sm text-gray-500">
            Error Code: <span class="text-gray-300 font-mono">403</span>
        </div>
    </div>
</body>
</html>