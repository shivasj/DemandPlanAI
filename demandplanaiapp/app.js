(function() {
    'use strict';
    angular.module('demandplanai.app', [
        'ui.router'
        ,'ui.bootstrap'
        ,'ngSanitize'
        ,'ngMessages'
        ,'LocalStorageModule'
        ,'ui.grid'
        ,'ui.grid.pinning'
        ,'ui.grid.exporter'
        ,'ui.grid.resizeColumns'
        ,'ui.grid.edit'
        ,'ui.grid.cellNav'
        ,'ui.grid.selection'
        ,'demandplanai.app.config'
        ,'demandplanai.app.common'
        ,'demandplanai.app.navbar'
        ,'demandplanai.app.menu'
        ,'demandplanai.app.home'
        ,'demandplanai.app.demandplan'
    ])
        .service('AppService',function($q,$timeout,$window,$state){
            var self = this;

            self.GetScreenWidth = function () {
                return $window.innerWidth;
            }
            self.IsMobile = function(){
                if($window.innerWidth < 700){
                    return true;
                }else{
                    return false;
                }
            }
            self.GetWindow = function () {
                return angular.element($window);
            }

            // Load External Libraries

        })
        .controller('AppController', function ($state,AppService) {
            var self = this;
            console.log("AppController");

        })

})();

