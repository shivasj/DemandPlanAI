(function() {
    'use strict';
    angular.module('demandplanai.app.home', [])
        .config(function ($stateProvider, $urlRouterProvider) {
            var now = new Date();
            var ticks = now.getTime();

            // Home
            $stateProvider.state('home', {
                url: '/',
                templateUrl: 'components/home/home.html?'+ticks,
                controller: 'HomeController',
                controllerAs: 'homeCtrl'
            });

            $urlRouterProvider.otherwise('/');

        })
        .service('DataService',function ($http, $q, $timeout,AppConfig){
            var self = this;

            var now = function () { return new Date(); };
            var ticks = now().getTime();
            var cache = {}

            // Get data
            self.GetCommodity = function () {
                var deferred = $q.defer();

                Papa.parse(AppConfig.Settings.py_service_url + '/commodity' , {
                    download: true,
                    header: true,
                    worker: true,
                    dynamicTyping:true,
                    complete: function(results, file) {
                        console.log("Parsing complete:", results, file);
                        deferred.resolve(results);
                    }
                })
                return deferred.promise;
            }
            self.GetDemandHistory = function () {
                var deferred = $q.defer();

                Papa.parse(AppConfig.Settings.py_service_url + '/demand_history' , {
                    download: true,
                    header: true,
                    worker: true,
                    dynamicTyping:true,
                    complete: function(results, file) {
                        console.log("Parsing complete:", results, file);
                        deferred.resolve(results);
                    }
                })
                return deferred.promise;
            }
            self.GetTopModels = function () {
                var deferred = $q.defer();

                Papa.parse(AppConfig.Settings.py_service_url + '/top_models' , {
                    download: true,
                    header: true,
                    worker: true,
                    dynamicTyping:true,
                    complete: function(results, file) {
                        console.log("Parsing complete:", results, file);
                        deferred.resolve(results);
                    }
                })
                return deferred.promise;
            }
            self.Predict = function (key,epochs,look_back,commodity) {
                return $http.get(AppConfig.Settings.py_service_url + '/predict?' + "key=" + key + "&epochs="+epochs+'&look_back='+look_back + '&commodity='+commodity, { cache: true });
            }

        })
        .controller('HomeController',function(DataService,AppService,$state, $stateParams){
            var self = this;

            $state.go('demandplan');
        })

})();